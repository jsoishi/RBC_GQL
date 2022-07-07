import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def parse_file(filename):
    sim_time = []
    dt = []
    KE = []
    with open(filename,'r') as df:
        for line in df:
            m1 = re.search("Time: ([0-9.e+-]*), dt: ([0-9.e+-]*)",line)
            if m1:
                sim_time.append(float(m1.group(1)))
                dt.append(float(m1.group(2)))
            m2 = re.search("avg KE = ([0-9.e+-]*)", line)
            if m2:
                KE.append(float(m2.group(1)))

    return np.array(sim_time), np.array(dt), np.array(KE)

curtis_fname = sys.argv[-2]
our_fname = sys.argv[-1]

c_t, c_dt, c_ke = parse_file(curtis_fname)
o_t, o_dt, o_ke = parse_file(our_fname)

plt.subplot(211)
#plt.plot(o_t, o_dt, 'x', label='mine')
plt.plot(c_t[:100], c_dt[:100]-o_dt[:100])#, label='curtis')
#plt.legend()
plt.xlabel("t")
plt.ylabel("dt")

plt.subplot(212)
#plt.plot(o_t, o_ke, 'x', label='mine')
plt.plot(c_t[:100], c_ke[:100] - o_ke[:100])#, label='curtis')
plt.xlabel("t")
plt.ylabel("KE")
plt.tight_layout()

plt.savefig('dt_ke.png',dpi=300)
