from pathlib import Path
p=Path('tools/apply_feynman_update.py')
s=p.read_text(encoding='utf-8')
a=s.index('# Verify extracted code corresponds to XML code paragraphs')
b=s.index("tree.write(doc,encoding='UTF-8',xml_declaration=True)")
rep="""# Live Editor stores the script inside one or a few code paragraph payloads\n# whose w:t text contains embedded newlines. Replace the primary payload.\nscript_paras=[q for q in root.iter(f'{{{W}}}p') if is_code(q)]\nif not script_paras:\n    raise RuntimeError('No MATLAB code paragraph found in Live Script XML.')\nprimary=None\nfor q in script_paras:\n    if '%% Demo: SINDy bypass' in ptext(q):\n        primary=q\n        break\nif primary is None:\n    primary=max(script_paras,key=lambda q:len(ptext(q)))\nsettext(primary,main)\nfor q in script_paras:\n    if q is not primary and ptext(q).strip():\n        parent[q].remove(q)\n"""
patched=s[:a]+rep+s[b:]
ns={'__name__':'__main__','__file__':str(p)}
exec(compile(patched,str(p),'exec'),ns,ns)
