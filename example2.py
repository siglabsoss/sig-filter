import sys
sys.path.insert(0,"../../python-osi")
from sigmath3 import *
# from subcarrier_math import *

from scipy.signal import savgol_filter

# from numpy.signal import savgol_filter

from numpy.fft import ifft, fft, fftshift


def _signed_value(value, bit_count):
    if value > 2**(bit_count - 1) - 1:
        value -= 2**bit_count 

    return value



def _load_hex_file(filepath):
    with open(filepath) as bootprogram:
        lines = bootprogram.readlines()

    words = [int(l,16) for l in lines]

    rf = []
    for w in words:
        real = _signed_value(w&0xffff, 16)
        imag = _signed_value((w >> 16) & 0xffff, 16)
        rf.append(np.complex(real,imag))
        # print "r", real, " i ", imag
        # print rf[0]
        # sys.exit(0)
    return rf


# in units of frames
samples = 60000

# delete from the front, in units of frames
start = 6000


# in units of sampels
rf = read_rf_grc("../data/grav3_dump0.raw", 64*samples);


off0 = 6

# downsample by 64, also adding a channel offset to extract samples from only 1 channel
slice0 = rf[off0::64]

# trim the channel from the front
slice0 = slice0[start:]



flen = 1021
forder = 1


filt0_r = savgol_filter(np.real(slice0), flen, forder, mode='mirror')
filt0_i = savgol_filter(np.imag(slice0), flen, forder, mode='mirror')

# filt1 = savgol_filter(np.real(slice0), 511, 2, mode='nearest')
# filt2 = savgol_filter(np.real(slice0), 511, 2, mode='constant')
# filt3 = savgol_filter(np.real(slice0), 511, 2, mode='wrap')

filt0 = [np.complex(filt0_r[i],filt0_i[i]) for i in range(len(filt0_r))]

# nplot(slice0, "real only")
# nplot(filt0, "f0", False)

# nplot(filt1, "f1", False)
# nplot(filt2, "f2", False) # weird up and down
# nplot(filt3, "f3", False)

if False:
	fname = "dump0_ch_6_flen_1010_o_1"
	save_rf_grc("../data/" + fname + "_filt.raw", filt0)
	save_rf_grc("../data/" + fname + "_src.raw", slice0)
	exit(0)


ncplot(filt0)
ncplot(slice0, "and filt", False)


nplotqam(slice0)
nplotqam(filt0, "and filt qam", False)

# nplot(np.angle(slice0), "a")
# ncplot(slice0, "complex")
# nplotqam(slice0)


# nplot(np.array(range(0,40)));

# ncplot(asfft, "As fft")
# ncplot(x, "orig")


nplotshow()











# xshort = x[256:1024+256]

# for i,y in enumerate(xshort):
#     print "i ", i, " y ", y

# save_rf_grc("cs10_out.raw", nocp)

# sc = 13
# sc = 5

# for idx in range(len(x)):
#     if idx % 64 == sc:
#         y.append(x[idx])

# # print(y)

# xx = []
# yy = []

# for p in y:
#     xx.append(np.real(p))
#     yy.append(np.imag(p))
