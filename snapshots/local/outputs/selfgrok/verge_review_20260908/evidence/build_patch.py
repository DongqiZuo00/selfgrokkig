import difflib,hashlib,json
from pathlib import Path
root=Path("outputs/selfgrok/verge_review_20260908")
base=Path("work/selfgrok/baseline")
parts=[]
changed=["src/common.py","src/verge_round_core.py","src/verge_repair_training.py"]
for f in changed+["src/test_verge_fresh_review.py"]:
 old=(base/f).read_text().splitlines(True) if (base/f).exists() else []
 new=(root/"review_copy"/f).read_text().splitlines(True)
 parts.extend(difflib.unified_diff(old,new,fromfile="a/"+f if old else "/dev/null",tofile="b/"+f))
(root/"patches/minimal_fixes.patch").write_text("".join(parts),newline="\n")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={"original_snapshot":sha(Path("work/selfgrok/engineering_snapshot.tar.gz")),
 "files":{f:{"before":sha(base/f),"after":sha(root/"review_copy"/f)} for f in changed},
 "note":"Source hashes prove only this audit's original-vs-review-copy identity, not model weight equivalence."}
(root/"evidence/source_manifest.json").write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))

