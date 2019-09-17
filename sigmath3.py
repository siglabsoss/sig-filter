import os
import collections
import csv
import difflib
import errno
import hashlib
import itertools
import logging
import math
import numpy as np
import random
import scipy
import socket
import string
import struct
import sys
import time
from itertools import chain, repeat
from numpy.fft import fft, fftshift
from scipy import ndimage
import numpy as np
import inspect
import matplotlib.pyplot as plt
# import pyximport;pyximport.install()
# import zmq

# from osibase import OsiBase
# from filterelement import FilterElement
# from sigmathcython import *

# converts string types to complex
def raw_to_complex(str):
    f1 = struct.unpack('%df' % 1, str[0:4])
    f2 = struct.unpack('%df' % 1, str[4:8])

    f1 = f1[0]
    f2 = f2[0]
    return f1 + f2*1j

def complex_to_raw(n):

    s1 = struct.pack('%df' % 1, np.real(n))
    s2 = struct.pack('%df' % 1, np.imag(n))

    return s1 + s2

# converts complex number to a pair of int16's, called ishort in gnuradio
def complex_to_ishort(c):
    short = 2**15-1
    re = struct.pack("h", np.real(c)*short)
    im = struct.pack("h", np.imag(c)*short)
    return re+im

# i first, q second
def complex_to_ishort_multi(floats, endian=""):
    rr = np.real(floats)
    ii = np.imag(floats)
    zz = np.array((rr,ii)).transpose()
    zzz = zz.reshape(len(floats)*2) * (2**15-1)
    bytes = struct.pack(endian+"%sh" % len(zzz), *zzz)
    return bytes

# q first, i second
def complex_to_ishort_multi_rev(floats, endian=""):
    rr = np.real(floats)
    ii = np.imag(floats)
    zz = np.array((ii,rr)).transpose()
    zzz = zz.reshape(len(floats)*2) * (2**15-1)
    bytes = struct.pack(endian+"%sh" % len(zzz), *zzz)
    return bytes

# i first, q second
def ishort_to_complex_multi(ishort_bytes, endian=""):
    packed = struct.unpack(endian+"%dh" % int(len(ishort_bytes)/2), ishort_bytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

# q first, i second
def ishort_to_complex_multi_rev(ishort_bytes, endian=""):
    packed = struct.unpack(endian+"%dh" % int(len(ishort_bytes)/2), ishort_bytes)
    rere = sig_everyn(packed, 2, 1)
    imim = sig_everyn(packed, 2, 0)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

def complex_to_raw_multi(floats, endian=""):
    rr = np.real(floats)
    ii = np.imag(floats)
    zz = np.array((rr,ii)).transpose()
    zzz = zz.reshape(len(floats)*2)
    bytes = struct.pack(endian+"%sf" % len(zzz), *zzz)
    return bytes

def raw_to_complex_multi(raw_bytes):
    packed = struct.unpack("%df" % int(len(raw_bytes)/4), raw_bytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

# a pretty-print for hex strings
# def get_rose(data):
#     try:
#         ret = ' '.join("{:02x}".format(ord(c)) for c in data)
#     except TypeError, e:
#         ret = str(data)
    # return ret

# def print_rose(data):
#     print get_rose(data)

# def get_rose_int(data):
#     # adding int values
#     try:
#         ret = ' '.join("{:02}".format(ord(c)) for c in data)
#     except TypeError, e:
#         ret = str(data)
#     return ret

# if you want to go from the pretty print version back to a string (this would not be used in production)
def reverse_rose(input):
    orig2 = ''.join(input.split(' '))
    orig = str(bytearray.fromhex(orig2))
    return orig

# this is meant to replace print comma like functionality for ben being lazy
def s_(*args):
    out = ''
    for arg in args:
        out += str(arg)+' '
    out = out[:-1]
    return out

# def print_hex(str, ascii = False):
#     print 'hex:'
#     tag = ''
#     for b in str:
#         if ascii:
#             if b in string.printable:
#                 tag = b
#             else:
#                 tag = '?'
#         print ' ', format(ord(b), '02x'), tag

# def print_dec(str):
#     print 'hex:'
#     for b in str:
#         print ' ', ord(b)

# http://stackoverflow.com/questions/10237926/convert-string-to-list-of-bits-and-viceversa
def str_to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def bits_to_str(bits):
    chars = []
    for b in range(len(bits) / 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def all_to_ascii(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(all_to_ascii, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(all_to_ascii, data))
    else:
        return data

def floats_to_bytes(rf):
    return ''.join([complex_to_ishort(x) for x in rf])

def bytes_to_floats(rxbytes):
    packed = struct.unpack("%dh" % int(len(rxbytes)/2), rxbytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    assert len(rere) == len(imim)
    rx = list(itertools.imap(np.complex, rere, imim))

# DO NOT USE ROUNDIN GERRORORROROROROOROROORORORO
def drange_DO_NOT_USE(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step



def drange_DO_NOT_USE2(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)

    return L


def drange(start, end=None, inc=None):
    """A range function, that does accept float increments..."""
    import math

    if end == None:
        end = start + 0.0
        start = 0.0
    else: start += 0.0 # force it to be a float

    if inc == None:
        inc = 1.0
    count = int(math.ceil((end - start) / inc))

    L = [None,] * count

    L[0] = start
    for i in xrange(1,count):
        L[i] = L[i-1] + inc
    return L




def unroll_angle(input):
    thresh = np.pi

    adjust = 0

    sz = len(input)

    output = [None]*sz

    output[0] = input[0]

    for index in range(1,sz):
        samp = input[index]
        prev = input[index-1]

        if(abs(samp-prev) > thresh):
            direction = 1
            if( samp > prev ):
                direction = -1
            adjust = adjust + 2*np.pi*direction

        output[index] = input[index] + adjust

    return output

def bits_cpm_range(bits):
    bits = [(b*2)-1 for b in bits] # convert to -1,1
    return bits

def bits_binary_range(bits):
    bits = [int((b+1)/2) for b in bits]  # convert to ints with range of 0,1
    return bits

def ip_to_str(address):
    return socket.inet_ntop(socket.AF_INET, address)


# def nonblock_read(sock, size):
#     try:
#         buf = sock.read()
#     except IOError, e:
#         err = e.args[0]
#         if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
#             return None  # No data available
#         else:
#             # a "real" error occurred
#             print e
#             sys.exit(1)
#     else:
#         if len(buf) == 0:
#             return None
#         return buf

# returns None if socket doesn't have any data, otherwise returns a list of bytes
# you need to set os.O_NONBLOCK on the socket at creation in order for this function to work
#   fcntl.fcntl(sock, fcntl.F_SETFL, os.O_NONBLOCK)
# def nonblock_socket(sock, size):
#     # this try block is the non blocking way to grab UDP bytes
#     try:
#         buf = sock.recv(size)
#     except socket.error, e:
#         err = e.args[0]
#         if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
#             return None  # No data available
#         else:
#             # a "real" error occurred
#             print e
#             sys.exit(1)
#     else:
#         # got data
#         return buf

# def nonblock_socket_from(sock, size):
#     # this try block is the non blocking way to grab UDP bytes
#     try:
#         buf, addr = sock.recvfrom(size)
#     except socket.error, e:
#         err = e.args[0]
#         if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
#             return None, None  # No data available
#         else:
#             # a "real" error occurred
#             print e
#             sys.exit(1)
#     else:
#         # got data
#         return buf, addr


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def save_rf_grc(filename, data):
    dumpfile = open(filename, 'wb')
    for s in data:
        dumpfile.write(complex_to_raw(s))
    dumpfile.close()

def read_rf_grc(filename, max_samples=None):
    file = open(filename, 'rb')
    piece_size = 8
    dout = []
    sample_count = 0
    while True:
        bytes = file.read(piece_size)

        if bytes == "" or bytes == b"":
            break  # end of file
        dout.append(raw_to_complex(bytes))
        sample_count += 1
        if max_samples is not None and sample_count >= max_samples:
            break

    return dout



# write rf to a csv file
# to read file from matlab
# k = csvread('Y:\home\ubuntu\python-osi\qam4.csv');
# kc = k(:,1) + k(:,2)*1j;
def save_rf(filename, data):
    dumpfile = open(filename, 'w')
    for s in data:
        print >> dumpfile, np.real(s), ',', np.imag(s)
    dumpfile.close()

# read rf from a csv file
# if your file is dc in matlab, run this:
#   dcs = [real(dc) imag(dc)];
#   csvwrite('filename.csv',dcs);
#   csvwrite('h3packetshift.csv', [real(dc) imag(dc)]);
def read_rf(filename):
    # read a CSV file
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',', quotechar='|')
    data = []
    for row in reader:
        data.append(float(row[0]) + float(row[1])*1j)
    file.close()
    return data

def read_rf_hack(filename):
    # read a CSV file
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',', quotechar='|')
    data = []
    for row in reader:
        data.append(float(row[0]))
        data.append(float(row[1]))
    file.close()
    return data

# basic logging setup
def setup_logger(that, name, prefix=None):
    if prefix is None:
        prefix = name

    that.log = logging.getLogger(name)
    that.log.setLevel(logging.INFO)
    # create console handler and set level to debug
    lch = logging.StreamHandler()
    lch.setLevel(logging.INFO)
    lfmt = logging.Formatter(prefix+': %(message)s')
    # add formatter to channel
    lch.setFormatter(lfmt)
    # add ch to logger
    that.log.addHandler(lch)

# convert a list of bits to an unsigned int
# h is the number of bits we are expecting in the list
def bit_list_unsigned_int(lin, h):
    sym = 0
    for j in range(h):
        try:
            sym += lin[j]*2**(h-j-1)
        except IndexError:
            sym += 0
    return sym

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return (idx,array[idx])




def tone_gen(samples, fs, hz):
    assert type(samples) == type(0)
    assert type(fs) in (int, long)

    inc = 1.0/fs * 2 * np.pi * hz

    if hz == 0:
        args = np.array([0] * samples)
    else:
        args = np.linspace(0, (samples-1) * inc, samples) * np.array([1j] * samples)
    return np.exp(args)











def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N



# ---------------- Beginning of matplotlib
##
# @defgroup nplot Plotting related helper functions
# @{
#

_nplot_figure = 0

##
#  Plot the sparsity pattern of 2-D array
# @param A the 2-D array
# @param title The plot title (string)
def nplotspy(A, title=""):
    fig = nplotfigure()

    plt.title(title)
    plt.spy(A, precision=0.01, markersize=3)

    return fig

def ncplot(rf, title=None, newfig=True):
    fig = None


    if newfig:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    plt.plot(np.real(rf))
    plt.plot(np.imag(rf), 'r')
    return fig

def nplot(rf, title=None, newfig=True):
    fig = None


    if newfig:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    plt.plot(np.real(rf))
    return fig

def nplotdots(rf, title="", hold=False):
    fig = None
    if hold:
        plt.hold(True)
    else:
        fig = nplotfigure()
        plt.title(title)
    plt.plot(range(len(rf)), np.real(rf), '-ko')
    return fig


def nplotqam(rf, title="", newfig=True):
    fig = None

    if newfig:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    plt.plot(np.real(rf), np.imag(rf), '.b', alpha=0.6)
    return fig


def nplotfftold(rf, title="", hold=False):
    fig = None

    if hold:
        plt.hold(True)
    else:
        fig = nplotfigure()
        plt.title(title)

    plt.plot(abs(fftshift(fft(rf))))
    return fig

def sig_peaks(bins, hz, num = 1, peaksHzSeparation = 1):
    binsabs = np.abs(bins)

    llen = len(bins)

    # approx method, assumes all bins are evenly spaces (they better be)
    binstep = hz[1] - hz[0]
    hzsepinbins = np.ceil(peaksHzSeparation / binstep)

    res = []

    for i in range(num):
        maxval,peakidx = sig_max(binsabs)
        res.append(peakidx)

        lowbin = int(np.max([0,peakidx-hzsepinbins]))
        highbin = int(np.min([llen,peakidx+hzsepinbins]))

        binsabs[lowbin:highbin] = 0.


    return res


def nplotfft(rf, fs = 1, title=None, newfig=True, peaks=False, peaksHzSeparation=1, peaksFloat=1.1):
    fig = None

    if newfig is True:
        fig = nplotfigure()

    if title is not None:
        plt.title(title)

    N = len(rf)

    X = fftshift(fft(rf)/N)
    absbins = 2*np.abs(X)
    df = fs/N
    f = np.linspace(-fs/2, fs/2-df, N)

    plt.plot(f, absbins)
    plt.xlabel('Frequency (in hertz)')
    plt.ylabel('Magnitude Response')


    if peaks:
        peaksres = sig_peaks(X, f, peaks, peaksHzSeparation)
        if type(newfig) is type(True):
            ax = fig.add_subplot(111)
        else:
            ax = newfig


        for pk in peaksres:

            maxidx = pk
            maxval = absbins[pk]

            lbl = s_('hz:', f[maxidx])

            ax.annotate(lbl, xy=(f[maxidx], maxval), xytext=(f[maxidx], maxval * peaksFloat),
                        arrowprops=dict(facecolor='black'),
                        )

    return fig


def nplothist(x, title = "", bins = False):
    fig = nplotfigure()

    plt.title(title)

    # more options
    # n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    if bins is not False:
        plt.hist(x, bins)
    else:
        plt.hist(x)

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    return fig


def nplotber(bers, ebn0, titles, title = ""):
    assert(len(bers) == len(ebn0) == len(titles))

    fig = nplotfigure()

    maintitle = "BER of ("

    for i in range(len(bers)):
        # check if any values are positive
        if any(k > 0 for k in bers[i]):
            # if so plot normally
            plt.semilogy(ebn0[i], bers[i], '-s', linewidth=2)
            maintitle += titles[i] + ', '
        else:
            # if all values are zero, don't plot (doing so forever prevents this plot window from drawing new lines)
            maintitle += 'ValueError: Data has no positive values, and therefore can not be log-scaled.' + ', '


    maintitle += ")"


    gridcolor = '#B0B0B0'
    plt.grid(b=True, which='major', color=gridcolor, linestyle='-')
    plt.grid(b=True, which='minor', color=gridcolor, linestyle='dotted')

    plt.legend(titles)
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('BER')

    if title == "":
        plt.title(maintitle)
    else:
        plt.title(title)

    return fig


def nplotmulti(xVals, yVals, legends, xLabel ='x', yLabel ='y', title ='', semilog = False, newfig=True, style='-s'):
    assert(len(yVals) == len(xVals) == len(legends))

    fig = None
    if newfig:
        fig = nplotfigure()
        plt.title(title)

    for i in range(len(yVals)):
        if semilog == True:
            plt.semilogy(xVals[i], yVals[i], style, linewidth=2)
        elif semilog == False:
            plt.plot(xVals[i], yVals[i], style, linewidth=2)

    gridcolor = '#B0B0B0'
    plt.grid(b=True, which='major', color=gridcolor, linestyle='-')
    plt.grid(b=True, which='minor', color=gridcolor, linestyle='dotted')

    plt.legend(legends)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)



    return fig


def nplotangle(angle, title = None, newfig=True):
    global _nplot_figure
    fig = None
    if newfig:
        fig = plt.figure(_nplot_figure, figsize=(7.,6.))
        _nplot_figure += 1

    res = 100
    cir = np.exp(1j*np.array(range(res+1))/(res/np.pi/2))
    cir = np.concatenate(([0],cir))

    cir = cir * np.exp(1j*angle)

    plt.plot(np.real(cir), np.imag(cir), 'b', alpha=0.6, linewidth=3.0)
    if title is not None:
        plt.title(title)
    else:
        plt.title(s_("angle:", angle))
    return fig


def nplotshow():
    plt.show()


def nplotfigure():
    global _nplot_figure

    fig = plt.figure(_nplot_figure)
    _nplot_figure += 1

    return fig


def nplotresponselinear(b, a, cutoff, fs):
    w, h = scipy.signal.freqz(b, a, worN=8000)
    nplotfigure()
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Linear Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()



## Plots frequency response of digital filter
# @param b b coefficients
# @param a a coefficients
# @param mode set mode 'frequency' and also pass in fs to display units in hz, or set mode 'radians'
# @param title graph title
# @param fs fs of filter in 'frequency' mode
# @param cutoff draw a vertial line at the filters cutoff if you know it
def nplotresponse(b, a, mode='frequency', title='', fs=0, cutoff=None):
    w, h = scipy.signal.freqz(b, a)

    fig = nplotfigure()
    plt.title(title)
    ax1 = fig.add_subplot(111)

    if mode == 'frequency':
        wplot = 0.5 * fs * w / np.pi
        xlab = 'Frequency [Hz]'
    elif mode == 'radians':
        wplot = w
        xlab = 'Frequency [rad/sample]'
    else:
        assert 0

    plt.plot(wplot, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel(xlab)
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(wplot, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')

    if cutoff is not None:
        plt.axvline(cutoff, color='k')

    return fig


def nplottext(str, newfig=True):
    fig = None
    if newfig == True:
        fig = nplotfigure()
        plt.title('txt')
        ax = fig.add_subplot(111)
    else:
        ax = newfig

    ax.set_aspect(1)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax.text(0, 0, str, ha="center", va="center", size=14,
            bbox=bbox_props)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    return fig


##
# @}
#



def rand_string_ascii(len):
    return ''.join(random.choice(string.ascii_letters) for _ in range(len))



def sigflatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def sigdup(listt, count=2):
    # return list(flatten([y for y in repeat([y for y in listt], count)]))
    return list(sigflatten([repeat(x, count) for x in listt]))



## wrapper around numpy's built in FFT that returns frequency of each bin
# note that this function returns data that has already been fftshift'd
# @param data data to fft
# @param fs samples per second to be used for bin calculation
# @returns (fft_data, bins)
def sig_fft(data, fs):

    dt = 1 / fs     #                % seconds per sample
    N = len(data)
     # Fourier Transform:
    X = fftshift(fft(data)/N)
    # %% Frequency specifications:
    dF = float(fs) / N     #                % hertz
    # f = -Fs/2:dF:Fs/2-dF;#           % hertz
    bins = drange(-fs / 2, fs / 2, dF)

    return X, bins
    # %% Plot the spectrum:
    # figure;
    # plot(f,2*abs(X));
    # xlabel('Frequency (in hertz)');
    # title('Magnitude Response');
    # disp('Each bin is ');
    # disp(dF);


def sig_rms(x):
    return np.sqrt(np.mean(x * np.conj(x)))



## Pass samples of a peak
# Will polyfit, and then return maximum of fit curve
# @param peak's data samples, ususally I pass 8
# @param order order of polyfit, i use 4
# @param doplot pass True for very nice plot of what's going on
def polyfit_peak_climb(data, order, doplot = False):
    # http://stackoverflow.com/questions/29634217/get-minimum-points-of-numpy-poly1d-curve
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html

    # assume that interp os over 0-len(data)
    llen = len(data)
    x = range(llen)
    p = np.polyfit(x, data, order)
    pp = np.poly1d(p)  # create an object for easy lookups

    if doplot:
        # dont need to do costly linspace
        xp = np.linspace(x[0], x[-1], 100)
        nplotfigure()
        plt.plot(x, data, '.', xp, pp(xp))



    # this takes the deriv and idk exactly how it works
    crit = pp.deriv().r
    r_crit = crit[crit.imag == 0].real
    test = pp.deriv(2)(r_crit)
    x_max = r_crit[test < 0]

    if doplot:
        y_min = pp(x_max)
        plt.plot(x_max, y_min, 'o')
        xc = np.arange(0, 7, 0.02)
        yc = pp(xc)
        plt.plot(xc, yc)

    # with super high orders sometimes two maxima can be found
    # this attempts to remove any maxima that are outside the area of interest
    if len(x_max) > 1:
        # these two lines make bool arrays for the given conditions
        validleft = x_max >= 0
        validright = x_max <= llen
        valid = []
        for j in range(len(x_max)):
            if validleft[j] and validright[j]:
                valid.append(x_max[j])

        assert len(valid) == 1, "More than one maxima found in range of interest"
        x_max = valid


    # due to weirdness above, always unencapsulate off the only list element
    return x_max[0]




# if os.environ.has_key("UNITTEST_NO_X11"):
#     def nplot_dummy(*args, **kwargs):
#         pass

#     nplotspy = nplot_dummy
#     nplot = nplot_dummy
#     nplotdots = nplot_dummy
#     nplotqam = nplot_dummy
#     nplotfftold = nplot_dummy
#     nplotfft = nplot_dummy
#     nplothist = nplot_dummy
#     nplotber = nplot_dummy
#     nplotmulti = nplot_dummy
#     nplotangle = nplot_dummy
#     nplotshow = nplot_dummy
#     nplotfigure = nplot_dummy
#     nplotresponselinear = nplot_dummy
#     nplotresponse = nplot_dummy
#     nplottext = nplot_dummy

