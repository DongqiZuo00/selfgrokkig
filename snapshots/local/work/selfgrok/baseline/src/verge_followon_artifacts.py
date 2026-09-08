"""Final all-candidate artifact gate. No draws, hashes or full-book completion."""
import argparse
import verge_followon_controller as ctl
import verge_followon_report as report
from verge_followon_csv_contract import COLUMNS,validate_csv


def validate_png(path,minimum_width,minimum_height):
    from PIL import Image
    with Image.open(path) as picture:
        if picture.format!='PNG' or picture.width<minimum_width or picture.height<minimum_height:
            raise RuntimeError('Invalid or undersized figure/preview')
        picture.verify()


def validate_visual_review(review,root):
    if (review.get('synthetic') is not False or review.get('main_figure_reviewed') is not True
            or set(review.get('csv_previews_reviewed',[]))!=set(COLUMNS)
            or len(review.get('csv_previews_reviewed',[]))!=7
            or review.get('all_labels_and_values_readable') is not True
            or review.get('latex_values_reviewed') is not True):
        raise RuntimeError('Final real-artifact visual/value review is missing')
    # File size and timestamp bind the review to the files actually viewed. No
    # content hashes or checkpoint inspection is involved. Record this AFTER
    # visual inspection; never synthesize a positive review from a renderer test.
    paths=['followon_summary.png','result_tables.tex']+[f'{n}.preview.png' for n in COLUMNS]
    entries=review.get('reviewed_file_metadata',{})
    if set(entries)!=set(paths):raise RuntimeError('Review is not tied to the complete current artifact set')
    for name in paths:
        stat=(root/name).stat()
        if entries[name]!={'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}:
            raise RuntimeError('Reviewed artifact changed; inspect its current version')


def validate():
    data,snapshot,hardware=report.collect()
    bundle=ctl.read(ctl.ROOT/'report_bundle.json')
    if (bundle.get('synthetic') is not False or bundle.get('analysis')!=data
            or bundle.get('hardware')!=hardware
            or bundle.get('resources',{}).get('allocations')!=snapshot['allocations']):
        raise RuntimeError('Synthetic, stale or incomplete follow-on report bundle')
    if (ctl.read(ctl.ROOT/'analysis.json')!=data
            or ctl.read(ctl.ROOT/'report_values.json')!=hardware
            or ctl.read(ctl.ROOT/'resource_accounting_final.json')!=bundle['resources']):
        raise RuntimeError('Standalone follow-on records differ from the validated bundle')
    result=validate_csv(bundle,ctl.ROOT)
    if (ctl.ROOT/'RESULTS_FOLLOWON.md').read_text(encoding='utf-8')!=report.report(data,hardware):
        raise RuntimeError('Markdown differs from current source values')
    if (ctl.ROOT/'result_tables.tex').read_text(encoding='utf-8')!=report.latex_tables(data,hardware):
        raise RuntimeError('LaTeX differs from current source values')
    validate_png(ctl.ROOT/'followon_summary.png',800,400)
    for name in COLUMNS:validate_png(ctl.ROOT/f'{name}.preview.png',400,80)
    validate_visual_review(ctl.read(ctl.ROOT/'artifact_visual_review.json'),ctl.ROOT)
    return dict(result,protocol='verge_followon_v1_artifact_validation',followon_block_complete=True,
        suite_complete=False,scope='Only equal-budget follow-ons of all 72 primary candidates',
        candidates=72,cohorts=24,loss_tokens=72*262144,official_test_opened=False,
        checkpoint_hash_scans=0,new_model_samples=0,required_matrix_remaining=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--save',action='store_true')
    args=parser.parse_args();result=validate()
    if args.save:ctl.write(ctl.ROOT/'FOLLOWON_BLOCK_COMPLETE.json',result)
    print(ctl.json.dumps(result,indent=2))
