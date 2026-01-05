import torch
torch.set_grad_enabled(False)

from enhancer import get_subject_isolation_pipeline

print("🔥 Preloading SubjectIsolationPipeline...")
get_subject_isolation_pipeline()
print("✅ Models preloaded")
