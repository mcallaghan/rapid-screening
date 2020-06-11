import re
from shutil import copyfile

with open("manuscript/main.tex") as f:
    pat = r"\\includegraphics\[width=.*\]{(.*)}"
    matches = re.finditer(pat, f.read())

    # need to also move those as they are regenerated
    
for m in matches:
	p = m.group(1)
	print(p)
	p = p.replace('2_figs_','images/')
	d = p.replace("images/","manuscript/2_figs_")
	print(f"copying {p} to {d}")
	try:
	    copyfile(p,d)
	    print("copied")
	    #break
	except:
	    pass


copyfile("mendeley.bib","manuscript/3_bib_mendeley.bib")
copyfile("tables/datasets.tex", "manuscript/1_tables_datasets.tex")
