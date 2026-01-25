import os
files=[]
root='.'
for dp, dn, fn in os.walk(root):
    for f in fn:
        p=os.path.join(dp,f)
        try:
            s=os.path.getsize(p)
            files.append((s,p))
        except Exception:
            pass
files.sort(reverse=True)
for s,p in files[:60]:
    print(f"{s/1e9:.2f} GB\t{s/1e6:.1f} MB\t{p}")
