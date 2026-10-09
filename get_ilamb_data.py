import getopt
import hashlib
import html
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


BASE = 'https://www.ilamb.org/CMIP6/historical/'
MAP_TYPES = {'timeint', 'bias', 'biasscore', 'rmse', 'rmsescore',
             'phase', 'shift', 'shiftscore'}


def key(value):
    return re.sub(r'[^a-z0-9]', '', value.casefold())


def matches(wanted, *values):
    return wanted is None or wanted.casefold() == 'all' or any(
        key(wanted) == key(value) for value in values)


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.rows, self.selects, self.images, self.links = [], {}, [], []
        self.row = self.cell = self.select = self.option = None
        self.feed(source)
        self.handle_endtag('tr')

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])
        if tag == 'tr':
            self.handle_endtag('td')
            self.handle_endtag('tr')
            self.row = {'classes': attrs.get('class', '').split(), 'cells': []}
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = {'text': '', 'links': []}
        elif tag == 'a' and self.cell is not None and 'href' in attrs:
            self.cell['links'].append(attrs['href'])
        elif tag == 'select':
            self.select = attrs.get('id', '')
            self.selects.setdefault(self.select, [])
        elif tag == 'option' and self.select is not None:
            self.option = {'value': attrs.get('value'), 'text': ''}
        elif tag == 'img':
            self.images.append(attrs)

    def handle_data(self, data):
        if self.cell is not None:
            self.cell['text'] += data
        if self.option is not None:
            self.option['text'] += data

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.cell['text'] = ' '.join(self.cell['text'].split())
            self.row['cells'].append(self.cell)
            self.cell = None
        elif tag in ('tr', 'tbody', 'table') and self.row is not None:
            self.handle_endtag('td')
            self.rows.append(self.row)
            self.row = self.cell = None
        elif tag == 'option' and self.option is not None:
            self.option['text'] = self.option['text'].strip()
            self.option['value'] = self.option['value'] or self.option['text']
            self.selects[self.select].append(self.option)
            self.option = None
        elif tag == 'select':
            self.select = None


def same_site(base, relative):
    url = urljoin(base, relative)
    if urlsplit(url).netloc != urlsplit(base).netloc:
        raise ValueError(f'Unexpected external link: {url}')
    return url


def datasets(source, base):
    category = subcategory = None
    result = {}
    for row in Page(source).rows:
        if not row['cells']:
            continue
        cell = row['cells'][0]
        if 'parent' in row['classes']:
            category, subcategory = cell['text'], None
        elif 'child_variable' in row['classes']:
            subcategory = cell['text']
        elif 'child_dataset' in row['classes'] and cell['links']:
            if not category or not subcategory:
                continue
            parts = urlsplit(same_site(base, cell['links'][0]))
            url = urlunsplit(parts._replace(query='', fragment=''))
            if category != 'Relationships':
                result.setdefault(url, dict(category=category, subcategory=subcategory,
                                            dataset=cell['text'], page_url=url))
    if not result:
        raise ValueError('No ILAMB dataset rows found; the report layout may differ.')
    return sorted(result.values(), key=lambda item: item['page_url'])


ASSIGNMENT = re.compile(
    r'''document\.getElementById\(['"]([^'"]+)['"]\)\.src\s*=\s*([^\n;]+)''')
TOKEN = re.compile(r'''\s*(?:'([^']*)'|"([^"]*)"|(MNAME|RNAME))\s*''')


def expand(expression, model, region):
    values = {'MNAME': model, 'RNAME': region}
    tokens = []
    position = 0
    while position < len(expression):
        token = TOKEN.match(expression, position)
        if not token:
            return None
        literal1, literal2, variable = token.groups()
        tokens.append(values[variable] if variable else
                      literal1 if literal1 is not None else literal2)
        position = token.end()
        if position == len(expression):
            break
        if expression[position] != '+':
            return None
        position += 1
    return ''.join(tokens)


def figures(source, dataset):
    page = Page(source)
    models = page.selects.get('MeanStateModel', [])
    regions = page.selects.get('MeanStateRegion', [])
    plots = {p['value']: p['text'] for p in page.selects.get('AllModelsPlot', [])}
    image_ids = {image.get('id') for image in page.images}
    sources = {image.get('src') for image in page.images}
    result = {}
    for image_id, expression in ASSIGNMENT.findall(source):
        benchmark = image_id.startswith('benchmark_')
        plot = image_id.removeprefix('benchmark_')
        if plot not in MAP_TYPES or image_id not in image_ids:
            continue
        legend = f'legend_{plot}.png'
        if legend not in sources:
            continue
        for model in ([{'value': 'Benchmark', 'text': 'Benchmark'}] if benchmark else models):
            for region in regions:
                filename = expand(expression.strip(), model['value'], region['value'])
                if not filename or not filename.endswith('.png'):
                    continue
                url = same_site(dataset['page_url'], filename)
                result[url] = dict(
                    **dataset, model=model['value'], region=region['value'],
                    region_label=region['text'], plot=plot,
                    plot_label=plots.get(plot, plot), image_url=url,
                    legend_url=same_site(dataset['page_url'], legend),
                    source_group=dataset['page_url'])
    return list(result.values())


def netcdf_url(source, item):
    suffix = '_' + item['model'] + '.nc'
    links = {same_site(item['page_url'], link) for link in Page(source).links
             if urlsplit(link).path.rsplit('/', 1)[-1].endswith(suffix)}
    if len(links) != 1:
        raise ValueError(f"Expected one NetCDF link for {item['model']}, found {len(links)}")
    return links.pop()


def download_netcdf(client, item, output, pages):
    page_url = item['page_url']
    if page_url not in pages:
        pages[page_url] = client.get(page_url).decode('utf-8')
    url = netcdf_url(pages[page_url], item)
    identifier = hashlib.sha256(url.encode()).hexdigest()[:20]
    path = output / 'data' / f'{identifier}.nc'
    path.parent.mkdir(exist_ok=True)
    checksum = client.netcdf(url, path)
    item.update(netcdf_url=url, netcdf_path=str(path.relative_to(output)),
                netcdf_sha256=checksum)


def add_netcdf(client, output):
    path = output / 'manifest.json'
    manifest = json.loads(path.read_text())
    pages = {}
    manifest['errors'] = [error for error in manifest.get('errors', [])
                          if error.get('stage') != 'netcdf']
    failures = 0
    for item in manifest['figures']:
        try:
            download_netcdf(client, item, output, pages)
            print(f"NetCDF saved: {item['dataset']} / {item['model']}", file=sys.stderr)
        except (OSError, ValueError) as error:
            failures += 1
            manifest['errors'].append(dict(url=item['image_url'], stage='netcdf', error=str(error)))
        write_json(path, manifest)
    print(f"NetCDF ready for {len(manifest['figures']) - failures}/{len(manifest['figures'])} maps.")
    return 1 if failures else 0


def align_arrays(output):
    try:
        import numpy as np
        from PIL import Image
        from netCDF4 import Dataset
    except ImportError as error:
        raise ValueError('--align requires: python3 -m pip install numpy pillow netCDF4') from error

    records = json.loads((output / 'manifest.json').read_text())['figures']
    if not records:
        raise ValueError('No figures to align.')
    images, values, metadata = [], [], []
    for index, item in enumerate(records):
        if item['region'] != 'global':
            raise ValueError('Alignment currently requires global maps; regional masks are not inferred.')
        if not item.get('netcdf_path'):
            raise ValueError(f"Missing NetCDF for {item['image_path']}; run --add-netcdf first.")
        with Image.open(output / item['image_path']) as source:
            map_image = source.convert('RGBA')
        with Image.open(output / item['legend_path']) as source:
            legend = source.convert('RGBA')
        width = max(map_image.width, legend.width)
        combined = Image.new('RGB', (width, map_image.height + legend.height), 'white')
        map_x, legend_x = (width - map_image.width) // 2, (width - legend.width) // 2
        combined.paste(map_image, (map_x, 0), map_image)
        combined.paste(legend, (legend_x, map_image.height), legend)
        images.append(np.asarray(combined))

        with Dataset(output / item['netcdf_path']) as nc:
            if 'MeanState' not in nc.groups:
                raise ValueError(f"Missing MeanState group: {item['netcdf_path']}")
            group = nc.groups['MeanState']
            prefix = 'timeint_of_' if item['plot'] == 'timeint' else item['plot'] + '_map_of_'
            names = [name for name in group.variables if name.startswith(prefix)]
            if len(names) != 1:
                raise ValueError(f"Expected one {prefix} variable in {item['netcdf_path']}; found {names}")
            variable = group.variables[names[0]]
            data = np.ma.asarray(variable[:], dtype=np.float64).filled(np.nan)
            if data.ndim not in (1, 2):
                raise ValueError(f'Expected station or grid data, got shape {data.shape}')
            coordinates = {}
            for name, coordinate in group.variables.items():
                if not (name.startswith(('lat', 'lon')) or name in variable.dimensions):
                    continue
                array = np.ma.asarray(coordinate[:], dtype=np.float64).filled(np.nan)
                coordinates[name] = dict(dimensions=list(coordinate.dimensions),
                                         units=getattr(coordinate, 'units', ''),
                                         values=np.where(np.isfinite(array), array, None).tolist())
            values.append(data.reshape(-1))
            metadata.append(dict(index=index, id=item['id'],
                                 image_path=item['image_path'], legend_path=item['legend_path'],
                                 netcdf_path=item['netcdf_path'], source_group=item.get('source_group'),
                                 dataset=item['dataset'], model=item['model'], plot=item['plot'],
                                 variable='MeanState/' + names[0], units=getattr(variable, 'units', ''),
                                 shape=list(data.shape), dimensions=list(variable.dimensions),
                                 length=data.size, coordinates=coordinates,
                                 image_shape=list(images[-1].shape),
                                 map_box=[map_x, 0, map_x + map_image.width, map_image.height],
                                 legend_box=[legend_x, map_image.height, legend_x + legend.width,
                                             map_image.height + legend.height]))

    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    image_array = np.full((len(images), height, width, 3), 255, dtype=np.uint8)
    numerical_array = np.full((len(values), max(value.size for value in values)), np.nan)
    for index, (image, value) in enumerate(zip(images, values)):
        image_array[index, :image.shape[0], :image.shape[1]] = image
        numerical_array[index, :value.size] = value
    for name, array in [('images', image_array), ('numerical', numerical_array)]:
        temporary = output / (name + '.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, array, allow_pickle=False)
        temporary.replace(output / (name + '.npy'))
    write_json(output / 'alignment.json', dict(
        format_version=1, alignment='manifest order; source arrays, not image-pixel registration',
        images=dict(path='images.npy', shape=list(image_array.shape), dtype='uint8',
                    layout='RGB; map above centered legend; white padding; boxes are [left, top, right, bottom]'),
        numerical=dict(path='numerical.npy', shape=list(numerical_array.shape), dtype='float64',
                       layout='C-order flattened native arrays; NaN for masked data and trailing padding; native units'),
        records=metadata))
    print(f'Saved {len(records)} aligned pairs: images.npy {image_array.shape}, '
          f'numerical.npy {numerical_array.shape}, and alignment.json')
    return 0


class Client:
    def __init__(self, delay):
        self.delay, self.last = delay, 0.0

    def get(self, url):
        for attempt in range(3):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            try:
                request = Request(url, headers={'User-Agent': 'ILAMB-map-research-scraper/1.0'})
                with urlopen(request, timeout=30) as response:
                    return response.read()
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)

    def netcdf(self, url, path):
        return self.asset(url, path, (b'CDF\x01', b'CDF\x02', b'CDF\x05',
                                      b'\x89HDF\r\n\x1a\n'), 'NetCDF')

    def png(self, url, path):
        return self.asset(url, path, (b'\x89PNG\r\n\x1a\n',), 'PNG')

    def asset(self, url, path, signatures, label):
        data = path.read_bytes() if path.exists() else self.get(url)
        if not data.startswith(signatures):
            raise ValueError(f'Expected {label}, received something else: {url}')
        if not path.exists():
            temporary = path.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.replace(path)
        return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def options(argv):
    long = ['list', 'category=', 'subcategory=', 'dataset=', 'model=', 'region=',
            'plot=', 'random', 'seed=', 'count=', 'output=', 'catalog=',
            'catalog-only', 'base-url=', 'delay=', 'add-netcdf', 'align', 'help']
    parsed, extra = getopt.gnu_getopt(argv, '', long)
    if extra:
        raise ValueError(f'Unexpected arguments: {extra}')
    args = {name[2:]: value if value else True for name, value in parsed}
    if 'seed' in args and 'random' not in args:
        raise ValueError('--seed requires --random')
    args['seed'] = int(args.get('seed', 42))
    args['count'] = int(args.get('count', 20))
    args['delay'] = float(args.get('delay', 0.2))
    args.setdefault('model', 'all')
    args.setdefault('region', 'global')
    args.setdefault('plot', 'timeint')
    if args['count'] <= 0 or not 0 <= args['delay'] <= 60:
        raise ValueError('--count must be positive; --delay must be between 0 and 60')
    if args['plot'] not in MAP_TYPES | {'all'}:
        raise ValueError(f'Unknown --plot; choose from {sorted(MAP_TYPES)} or all')
    return args


def selected(items, args, fields):
    return [item for item in items if all(
        matches(args.get(field), item[field],
                item.get('region_label', '') if field == 'region' else item[field])
        for field in fields)]


def main(argv=None):
    args = options(sys.argv[1:] if argv is None else argv)
    if 'help' in args:
        print('Usage: python3 get_ilamb_data.py [options]\n'
              '  --list --category NAME --subcategory NAME --dataset NAME\n'
              '  --model NAME --region NAME --plot TYPE\n'
              '  --random --seed N --count N --output DIR\n'
              '  --catalog FILE --catalog-only --base-url URL --delay SECONDS\n'
              '  --add-netcdf  Add NetCDF files to the existing output manifest.\n'
              '  --align       Export existing map/legend pairs and numerical arrays to NumPy.\n'
              'New downloads include maps, legends, and linked NetCDF files.')
        return 0
    if 'align' in args:
        if 'add-netcdf' in args:
            raise ValueError('Run --add-netcdf before --align, as separate commands.')
        return align_arrays(Path(args.get('output', 'ilamb_maps')))
    client = Client(args['delay'])
    if 'add-netcdf' in args:
        return add_netcdf(client, Path(args.get('output', 'ilamb_maps')))
    pages = {}
    base = args.get('base-url', BASE).rstrip('/') + '/'
    errors = []
    if 'catalog' in args:
        catalog = json.loads(Path(args['catalog']).read_text())
        candidates = catalog['figures']
        entries = catalog['datasets']
        base = catalog['base_url']
    else:
        source = client.get(base).decode('utf-8')
        entries = datasets(source, base)
        candidates = None
    entries = selected(entries, args, ('category', 'subcategory', 'dataset'))
    if not entries:
        raise ValueError('No matching datasets. Use --list to see exact names.')
    if 'list' in args:
        for item in entries:
            print(' / '.join(item[field] for field in ('category', 'subcategory', 'dataset')))
        return 0
    output = Path(args.get('output', 'ilamb_maps'))
    output.mkdir(parents=True, exist_ok=True)
    if candidates is None:
        candidates = []
        for index, item in enumerate(entries, 1):
            print(f"Inspecting {index}/{len(entries)}: {item['subcategory']} / {item['dataset']}", file=sys.stderr)
            try:
                source = client.get(item['page_url']).decode('utf-8')
                pages[item['page_url']] = source
                candidates.extend(figures(source, item))
            except (OSError, ValueError) as error:
                errors.append({'url': item['page_url'], 'stage': 'discovery', 'error': str(error)})
        catalog = dict(base_url=base, created_utc=datetime.now(timezone.utc).isoformat(),
                       datasets=entries, figures=sorted(candidates, key=lambda x: x['image_url']),
                       discovery_errors=errors)
        write_json(output / 'catalog.json', catalog)
    else:
        errors.extend(catalog.get('discovery_errors', []))
    candidates = selected(candidates, args,
                          ('category', 'subcategory', 'dataset', 'model', 'region', 'plot'))
    candidates = sorted({item['image_url']: item for item in candidates}.values(),
                        key=lambda item: item['image_url'])
    if not candidates:
        raise ValueError('No supported map candidates match. Check filters or try --plot all.')
    digest = hashlib.sha256('\n'.join(x['image_url'] for x in candidates).encode()).hexdigest()
    print(f'{len(candidates)} eligible map candidates.', file=sys.stderr)
    if 'random' in args:
        random.Random(args['seed']).shuffle(candidates)
    manifest = dict(base_url=base, created_utc=datetime.now(timezone.utc).isoformat(),
                    options=args, candidate_pool_sha256=digest, candidate_count=len(candidates),
                    selection_order=[x['image_url'] for x in candidates],
                    figures=[], errors=errors)
    if 'catalog-only' in args:
        write_json(output / 'manifest.json', manifest)
        print(f'Catalog ready: {output.resolve()}')
        return 1 if errors else 0
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)
    for item in candidates:
        identifier = hashlib.sha256(item['image_url'].encode()).hexdigest()[:20]
        legend_id = hashlib.sha256(item['legend_url'].encode()).hexdigest()[:20]
        map_path, legend_path = assets / f'{identifier}.png', assets / f'legend_{legend_id}.png'
        try:
            map_hash = client.png(item['image_url'], map_path)
            legend_hash = client.png(item['legend_url'], legend_path)
            record = dict(item, id=identifier, image_path=str(map_path.relative_to(output)),
                          legend_path=str(legend_path.relative_to(output)),
                          image_sha256=map_hash, legend_sha256=legend_hash)
            download_netcdf(client, record, output, pages)
            manifest['figures'].append(record)
            print(f"Saved {len(manifest['figures'])}/{args['count']}: {item['dataset']} / {item['model']} / {item['plot']}", file=sys.stderr)
        except (OSError, ValueError) as error:
            manifest['errors'].append({'url': item['image_url'], 'stage': 'download', 'error': str(error)})
        write_json(output / 'manifest.json', manifest)
        if len(manifest['figures']) >= args['count']:
            break
    cards = []
    for item in manifest['figures']:
        title = ' / '.join(item[field] for field in ('category', 'subcategory', 'dataset', 'model', 'region', 'plot'))
        cards.append(f'<article><h2>{html.escape(title)}</h2>'
                     f'<img src="{html.escape(item["image_path"], quote=True)}">'
                     f'<br><img src="{html.escape(item["legend_path"], quote=True)}">'
                     f'<p><a href="{html.escape(item["page_url"], quote=True)}">Source</a> · '
                     f'<a href="{html.escape(item["netcdf_path"], quote=True)}">NetCDF</a></p></article>')
    (output / 'preview.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>ILAMB maps</title><style>body{font:16px sans-serif;margin:32px}'
        'img{max-width:100%}article{margin-bottom:48px}h2{font-size:18px}</style>' + '\n'.join(cards))
    count = len(manifest['figures'])
    print(f'Saved {count} map/legend pairs to {output.resolve()}')
    if manifest['errors']:
        print(f"{len(manifest['errors'])} failures recorded in manifest.json.", file=sys.stderr)
    if count < args['count']:
        print(f"Only {count} available pairs; requested {args['count']}.", file=sys.stderr)
    return 0 if count == args['count'] and not errors else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, getopt.GetoptError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(2)
