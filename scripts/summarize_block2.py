import json,glob,os,sys
root=sys.argv[1] if len(sys.argv)>1 else '/workspace/ic013_data/block2'
out={}
for f in sorted(glob.glob(f'{root}/*/salient_segmentation_results.json')):
    r=json.load(open(f)); n=os.path.basename(os.path.dirname(f))
    out[n]={k:{m:round(100*v,3) for m,v in r[k].items()} for k in ['Tokencut','Tokencut_BS','Tokencut_BS_binary']}
    out[n]['args']={k:r['args'][k] for k in ['backbone','features','semantic_invariance_projector','pca_guide_dim','flip_guide','mu','tau','gamma']}
    t=out[n]; print(f"{n:22s} noBS {t['Tokencut']['F_max']:.2f}/{t['Tokencut']['IoU']:.2f}  BS {t['Tokencut_BS']['F_max']:.2f}/{t['Tokencut_BS']['IoU']:.2f}/{t['Tokencut_BS']['accuracy']:.2f}  BSbin {t['Tokencut_BS_binary']['F_max']:.2f}/{t['Tokencut_BS_binary']['IoU']:.2f}")
json.dump(out,open('/workspace/ic013/results/block2_ecssd.json','w'),indent=1)
