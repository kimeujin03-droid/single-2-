import pandas as pd

p = r"C:\Users\kimeu\Downloads\Star_Hz_v2_12\Star_Hz\HERA_04-03-2022_all.pkl"
obj = pd.read_pickle(p)
print("type:", type(obj))

if isinstance(obj, (list, tuple)):
    print("len:", len(obj))
    if len(obj):
        first = obj[0]
        print("first type:", type(first))
        print("first shape:", getattr(first, "shape", None))
        if hasattr(first, "keys"):
            try:
                print("first keys sample:", list(first.keys())[:20])
            except Exception as e:
                print("keys inspect error:", e)
elif isinstance(obj, dict):
    print("keys sample:", list(obj.keys())[:50])
    for k in list(obj.keys())[:10]:
        v = obj[k]
        print(f"  key={k!r} -> type={type(v)}, shape={getattr(v,'shape',None)}")
else:
    print("shape:", getattr(obj, "shape", None))
    if hasattr(obj, "head"):
        try:
            print(obj.head())
        except Exception as e:
            print("head error:", e)
