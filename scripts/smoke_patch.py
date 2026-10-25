"""Day-0 smoke test: make CPU-runnable copies of SOAP's fit_projector.py / main_salient.py.
Upstream stubs get_data() with `...` and hard-codes CUDA; we patch only those lines.
Usage: python smoke_patch.py /workspace/soap   -> writes fit_projector_cpu.py, main_salient_cpu.py there."""
import sys, os, re
root = sys.argv[1]
GETDATA_IN = '''def get_data(dataset, datafolder, imgproc):
    import glob, PIL.Image as Image
    files = sorted(glob.glob(os.path.join(datafolder, '**', '*.JPEG'), recursive=True))
    files = files[::max(1, len(files)//int(os.environ.get('SMOKE_N', '64')))][:int(os.environ.get('SMOKE_N', '64'))]
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(files)
        def __getitem__(s, i): return imgproc(Image.open(files[i]).convert('RGB'))
    return DS()
'''
s = open(os.path.join(root, 'fit_projector.py')).read()
s = re.sub(r"def get_data\(dataset, datafolder, imgproc\):\n.*?return data\n", GETDATA_IN, s, flags=re.S)
s = s.replace("torch.device('cuda')", "torch.device('cpu')").replace("pin_memory=True", "pin_memory=False")
open(os.path.join(root, 'fit_projector_cpu.py'), 'w').write(s)

GETDATA_EC = '''def get_data(dataset:str, datafolder:str):
    import glob, PIL.Image as Image
    n = int(os.environ.get('SMOKE_N', '1000'))
    ims = sorted(glob.glob(os.path.join(datafolder, 'images', '*.jpg')))[:n]
    out = []
    for p in ims:
        g = os.path.join(datafolder, 'ground_truth_mask', os.path.basename(p)[:-4] + '.png')
        out.append((Image.open(p).convert('RGB'), (np.array(Image.open(g).convert('L')) > 127).astype(np.float32)))
    return out
'''
s = open(os.path.join(root, 'main_salient.py')).read()
s = re.sub(r"def get_data\(dataset:str, datafolder:str\):\n.*?raise NotImplementedError\(f'Dataset \{dataset\} is unknown'\)\n", GETDATA_EC, s, flags=re.S)
s = s.replace('.cuda()', '.cpu()')
open(os.path.join(root, 'main_salient_cpu.py'), 'w').write(s)
print('ok')
# salient/ncut.py also hard-codes .to('cuda'); make a CPU copy package salient_cpu/
import shutil
dst = os.path.join(root, 'salient_cpu')
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(os.path.join(root, 'salient'), dst, ignore=shutil.ignore_patterns('__pycache__'))
p = os.path.join(dst, 'ncut.py'); t = open(p).read().replace(".to('cuda')", ".to('cpu')"); open(p, 'w').write(t)
p = os.path.join(root, 'main_salient_cpu.py'); t = open(p).read().replace('from salient.', 'from salient_cpu.'); open(p, 'w').write(t)
