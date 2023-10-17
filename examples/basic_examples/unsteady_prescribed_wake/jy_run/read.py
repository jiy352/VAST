import numpy as np
import os

current = 0
snopt_files = [f for f in os.listdir('.') if f.endswith('SNOPT_summary.out')]
if len(snopt_files) != 1:
    raise ValueError('There should be only one SNOPT_summary.out file in the current directory')

with open(snopt_files[0], 'r') as f:
    lines=f.readlines()
lines.reverse()
rev_lines = lines

for i, line in enumerate(rev_lines, 0):
    if ('SNOPTC EXIT' in line):
        current = i + 2
        line = rev_lines[current]
        major = int(line[:6].strip())
        break
rev_opt  = []
rev_feas = []
rev_major= []
rev_nnf  = []
rev_nsup = []
rev_merit = []
rev_pen = []

while (major > 0):
    line  = rev_lines[current]
    major = int(line[:6].strip())
    rev_major.append(major)
    rev_nnf.append(int(line[25:29]))
    rev_feas.append(float(line[31:38]))
    rev_opt.append(float(line[40:47]))
    rev_merit.append(float(line[49:62]))
    # rev_nsup.append(int(line[62:69]))
    # pen = line[70:77]
    # print(len(pen))
    # if pen in (' '*7, '') :
    #     rev_pen.append(0.)
    # else:
    #     rev_pen.append(float(line[70:77]))
    current = current + 1
    if major%10 == 0:
        current = current + 2

rev_major.reverse()
major = np.array(rev_major)
rev_feas.reverse()
feas = np.array(rev_feas)
rev_opt.reverse()
opt = np.array(rev_opt)
rev_merit.reverse()
merit = np.array(rev_merit)
rev_nsup.reverse()
# nsup = np.array(rev_nsup)
# rev_pen.reverse()
# pen = np.array(rev_pen)
rev_nnf.reverse()
nnf = np.array(rev_nnf)
nnf = np.append([1,], np.ediff1d(nnf))

print(major)
print(feas)
print(opt)


import matplotlib.pyplot as plt
plt.figure(figsize=(5,4*5/6))

plt.semilogy(major, feas, '.-')
plt.semilogy(major, opt, '.-')
plt.xlabel('Major Iterations')
plt.ylabel('Feasibility and Optimality')    
plt.legend(['Feasibility', 'Optimality'])
# plt.title('Feasibility vs Major Iterations (log scale)')
plt.tight_layout()
plt.savefig('feas_opt.pdf',dpi=400,transparent=True)
plt.show()
