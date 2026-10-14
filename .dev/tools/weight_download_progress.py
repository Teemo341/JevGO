import datetime as dt
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
now = dt.datetime.now(dt.timezone.utc)
reports = []
for path in sorted((root / 'models/.download-cache/xet/logs').glob('*.log'))[-4:]:
    observations = []
    warnings = 0
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        warnings += item.get('level') in ('WARN', 'ERROR')
        match = re.search(r'observed bytes sent so far = (\d+)', item.get('fields', {}).get('message', ''))
        if match:
            observations.append((dt.datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')), int(match[1])))
    if observations:
        end, size = observations[-1]
        before = [item for item in observations if (end-item[0]).total_seconds() >= 60]
        start, old_size = before[-1] if before else observations[0]
        elapsed = (end-start).total_seconds()
        reports.append({'process':path.stem.split('_')[-1], 'network_GB':round(size/1e9,2),
                        'recent_MB_s':round((size-old_size)/elapsed/1e6,2) if elapsed and (now-end).total_seconds() < 60 else 0,
                        'last_event_seconds_ago':round((now-end).total_seconds()), 'warnings_errors':warnings})
manifest = json.loads((root/'configs/weights-manifest.json').read_text())
files = []
for model in manifest['models']:
    present = []
    for entry in model['files']:
        path = root/model['local_directory']/entry['path']
        if path.is_file() and path.stat().st_size == entry['size']:
            present.append(entry)
    files.append({'id':model['id'],'complete_size_files':len(present),'total_files':len(model['files']),
                  'complete_size_GB':round(sum(entry['size'] for entry in present)/1e9,2)})
print(json.dumps({'network':reports,'files':files},ensure_ascii=False,indent=2))
