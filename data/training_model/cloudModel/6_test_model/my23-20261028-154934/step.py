from collections import Counter
import json
from pathlib import Path
import tempfile
import numpy as np
import onnxruntime as ort
from sklearn.metrics import balanced_accuracy_score
from common import get, ids, json_file, sha, tsv, restore_bundle, publish_bundle
from export_onnx import export, parity
from model_io import load_bundle, matrix
from population_parser import read_panel


def execute(ctx):
    with tempfile.TemporaryDirectory(prefix='test-model-') as d:
        tmp=Path(d)
        if restore_bundle(ctx,tmp):
            return
        bundle,checksums=load_bundle(ctx,tmp)
        pipe,markers,classes=bundle['pipeline'],bundle['markers'],bundle['classes']
        keep=get(ctx.result(2,'test501.keep'),tmp/'test.keep')
        people=ids(keep)
        if len(people)!=501:
            raise ValueError('Exactly 501 test people are required')
        panel=read_panel(ctx.input('panel',tmp/'population.panel'))
        x=matrix(ctx,people,keep,markers,tmp)
        actual=[panel[s][1] for s in people]
        predicted=pipe.predict(x)
        probs=pipe.predict_proba(x)
        rows=[{'IID':s,'population':panel[s][0],'actual_group':truth,'predicted_group':str(pred),
               'correct':int(truth==pred),**{f'prob_{c}':float(p) for c,p in zip(classes,prob)}}
              for s,truth,pred,prob in zip(people,actual,predicted,probs)]
        tsv(tmp/'predictions.tsv',['IID','population','actual_group','predicted_group','correct',*[f'prob_{c}' for c in classes]],rows)
        tsv(tmp/'confusion_matrix.tsv',['actual_group',*classes],[{'actual_group':c,**{k:sum(r['actual_group']==c and r['predicted_group']==k for r in rows) for k in classes}} for c in classes])
        for field in ['actual_group','population']:
            counts=[]
            for k in sorted({r[field] for r in rows}):
                part=[r for r in rows if r[field]==k]
                counts.append({field:k,'people':len(part),'correct':sum(r['correct'] for r in part),
                               'accuracy':sum(r['correct'] for r in part)/len(part)})
            tsv(tmp/f'by_{field}.tsv',[field,'people','correct','accuracy'],counts)
        folder=tmp/'browser'
        export(pipe,markers,folder)
        session=ort.InferenceSession(str(folder/'model.onnx'),providers=['CPUExecutionProvider'])
        parity_report=parity(pipe,session,x,classes,ctx.cfg['onnx_probability_atol'])
        missing_test=int(np.isnan(x).sum())
        report={'training_people':2003,'test_people':501,'correct':sum(r['correct'] for r in rows),
                'accuracy':sum(r['correct'] for r in rows)/501,
                'balanced_accuracy':float(balanced_accuracy_score(actual,predicted)),
                'selected_markers':len(markers),'missing_test_genotypes':missing_test,
                'model_sha256':checksums['model_sha256'], 'run_id':ctx.run_id,
                'note':'Model is frozen. All 501 test people are retained. This is a rerun of an already examined reference split, not a new external validation.'}
        json_file(report,tmp/'test_summary.json')
        json_file(parity_report,tmp/'onnx_parity.json')
        metadata={'schema_version':1,'model_name':'my23_v5_ancestry_model','run_id':ctx.run_id,
                  'classes':classes,'feature_count':len(markers),'training_people':2003,'genome_build':'37',
                  'input_name':'genotypes','probabilities_output':'probabilities',
                  'model_sha256':sha(folder/'model.onnx'),'markers_sha256':sha(folder/'markers.json'),
                  'minimum_called_fraction':ctx.cfg['minimum_called_fraction'],
                  'maximum_incompatible_fraction':ctx.cfg['maximum_incompatible_fraction'],
                  'interpretation':'Broad reference-group classification; probabilities are not ancestry mixture percentages or clinical risk.'}
        json_file(metadata,folder/'metadata.json')
        files={name:tmp/name for name in ['predictions.tsv','confusion_matrix.tsv','by_actual_group.tsv','by_population.tsv','test_summary.json','onnx_parity.json']}
        files.update({f'browser/{name}':folder/name for name in ['model.onnx','markers.json','metadata.json']})
        publish_bundle(ctx,tmp,files)
        del x
