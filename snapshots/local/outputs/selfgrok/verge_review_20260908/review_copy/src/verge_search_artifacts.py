"""Final six-round search artifact gate, never whole-book completion."""
import argparse
import json
import verge_search_controller as ctl
import verge_search_report as report
from verge_search_csv_contract import COLUMNS,validate_csv
from verge_followon_artifacts import validate_png


def validate_visual_review(review,root):
    if (review.get('synthetic') is not False or review.get('main_figure_reviewed') is not True
            or set(review.get('csv_previews_reviewed',[]))!=set(COLUMNS)
            or len(review.get('csv_previews_reviewed',[]))!=6
            or review.get('all_labels_and_values_readable') is not True
            or review.get('latex_values_reviewed') is not True):
        raise RuntimeError('Current real-artifact visual review is required')
    names=['search_trajectories.png','result_table.tex']+[f'{n}.preview.png' for n in COLUMNS]
    metadata=review.get('reviewed_file_metadata',{})
    if set(metadata)!=set(names):raise RuntimeError('Review omits current artifact metadata')
    for name in names:
        stat=(root/name).stat()
        if metadata[name]!={'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}:
            raise RuntimeError('Reviewed artifact changed; inspect its current version')


def validate():
    current=report.inputs.collect();bundle=ctl.read(ctl.ROOT/'report_bundle.json')
    if (bundle.get('synthetic') is not False or bundle.get('analysis')!=current['analysis']
            or bundle.get('hardware')!=current['hardware']
            or bundle.get('resources',{}).get('allocations')!=current['resources']['allocations']):
        raise RuntimeError('Synthetic, stale or incomplete search report bundle')
    if ctl.read(ctl.ROOT/'render_values.json')!={'rounds':report.summary_rows(bundle),'synthetic':False,'suite_complete':False}:
        raise RuntimeError('Rendered values differ from current fixed endpoints')
    receipt=ctl.read(ctl.ROOT/'core_render_receipt.json')
    if any(receipt.get(k) is not v for k,v in {'synthetic':False,'figure_rendered':True,'markdown_rendered':True,
            'latex_rendered':True,'search_block_complete':False,'suite_complete':False}.items()):
        raise RuntimeError('Incomplete or foreign core render receipt')
    result=validate_csv(bundle,ctl.ROOT)
    for name,expected in [('RESULTS_SEARCH.md',report.markdown(bundle)),('result_table.tex',report.latex_table(bundle))]:
        if (ctl.ROOT/name).read_text(encoding='utf-8')!=expected:raise RuntimeError('Markdown or LaTeX differs from source values')
    validate_png(ctl.ROOT/'search_trajectories.png',800,400)
    for name in COLUMNS:validate_png(ctl.ROOT/f'{name}.preview.png',400,80)
    validate_visual_review(ctl.read(ctl.ROOT/'artifact_visual_review.json'),ctl.ROOT)
    return dict(result,protocol='verge_search_v1_artifact_validation',search_block_complete=True,suite_complete=False,
        scope='Only six rounds of four target-only search branches',rounds=6,branches=24,loss_tokens=12582912,
        official_test_opened=False,checkpoint_hash_scans=0,new_model_samples=0,required_matrix_remaining=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save',action='store_true');args=p.parse_args()
    result=validate()
    if args.save:ctl.write(ctl.ROOT/'SEARCH_BLOCK_COMPLETE.json',result)
    print(json.dumps(result,indent=2))
