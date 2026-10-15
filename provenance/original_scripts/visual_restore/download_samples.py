from pathlib import Path
import json,hashlib,urllib.request,urllib.parse
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'sample_selection.json').read_text())
for row in rows:
    url='https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/'+urllib.parse.quote(row['repo_path'],safe='/')
    path=ROOT/(row['panel']+'.jpg')
    content=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=45).read()
    assert hashlib.sha256(content).hexdigest()==row['sha256'],f'Image bytes do not match experiment: {row["panel"]}'
    path.write_bytes(content);row['download_url']=url;row['local_file']=path.name
    print('SHA256 PASS',row['panel'],flush=True)
(ROOT/'sample_selection.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
