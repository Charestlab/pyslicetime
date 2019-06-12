import numpy as np
from scipy.interpolate import pchip


def tseriesinterp(m, trorig, trnew, dim=None, numsamples=None,
                  fakeout=0, wantreplicate=False, interpmethod='pchip'):
    """Use interp1 to interpolate < m > (with extrapolation) such that
       the new version of < m > coincides with the original version of < m >
       at the first time point.  (If < fakeout > is used, the new version
       of < m > is actually shifted by < fakeout > seconds earlier than the
       original version of < m > .)

       Note that < m > can be complex-valued
       the real and imaginary parts are separately analyzed. This inherits
       from interp1's behavior.

    Args:
        m ([type]): < m > is a matrix with time-series
            data along some dimension. can also be a cell
            vector of things like that.
        trorig ([type]): < trorig > is the sampling
            time of < m > (e.g. 1 second)
        trnew ([type]): < trnew > is the new desired sampling time
        dim ([type]): < dim > (optional) is the
            dimension of < m > with time-series data. default to 2 if
            < m > is a row vector and to 1 otherwise.
        numsamples ([type]): < numsamples > (optional) is the number of
            desired samples. default to the number of samples
            that makes the duration of the new data match or
            minimally exceed the duration of the original data.
        fakeout ([type]): < fakeout > (optional) is a duration in seconds.
            If supplied, we act as if the time-series data was delayed
            by < fakeout >, and we obtain time points that correspond to
            going back in time by < fakeout > seconds.  Default: False.
        wantreplicate ([type]): < wantreplicate > (optional) is whether to
            repeat the first and last data points 3 times
            (e.g. 1 1 1 1 2 3 4 ...) before performing interpolation.
            The rationale is to try to avoid crazy extrapolation values.
            Default: False.
        interpmethod ([type]): < interpmethod > (optional) is the
            interpolation method, like 'pchip'. Default: 'pchip'.

    Returns < m > interpolated time-series.

    Example:
        import numpy as np
        from matplotlib import pyplot as plt
        x0 = np.arange(0.0, 10.1, 0.1)
        y0 = np.sin(x0)
        y1 = tseriesinterp(y0, .1, .23);
        plt.plot(x0, y0, 'r.-');
        plt.plot(0: .23: .23*(length(y1)-1), y1, 'go');

    Another example (complex data):
        import numpy as np
        from matplotlib import pyplot as plt
        x = (np.random.rand(1, 100)*2*pi)/4 + pi;
        x2 = ang2complex(x)
        y = tseriesinterp(x2, 1, .1, [], [], [], 1);
        y2 = mod(angle(y), 2*pi);

        plt.plot(list(range(x.shape[1])), x, 'ro');
        plot(linspacefixeddiff(1, .1, length(y2)), y2, 'b-');

    """

    # internal constants
    numchunks = 20

    # input
    if dim is None:
        dim = choose(isrowvector(m), 1, 0)

    if numsamples is None:
        numsamples = []

    # do it
    pnew = []
    for p in m:

        # prep 2D
        msize = p.shape[0]
        p = reshape2D(p, dim)

        # calc
        if numsamples is None:
            numsamples = np.ceil((p.shape[0]*trorig)/trnew)

        """
        f = linspacefixeddiff(x,d,n)
        x2 = x+d*(n-1);
        f = linspace(x,x2,n);
        """

        # do it
        if wantreplicate:

            pre = np.array((-3, -2, -1))*trorig
            meat = np.arange(0, trorig*p.shape[0], trorig)
            cherry = np.asarray(trorig*p.shape[0])
            dessert = p.shape[0]*trorig+np.array((1, 2, 3))*trorig

            timeorig = np.r_[pre, meat, cherry, dessert]

        else:
            timeorig = np.r_[
                np.arange(0, trorig*p.shape[0], trorig), trorig*p.shape[0]]

        timenew = np.arange(0, trnew*numsamples, trnew) - fakeout

        # do in chunks
        chunks = chunking(
            list(range(p.shape[1])), np.ceil(p.shape[1]/numchunks))
        temp = []
        mtemp = p

        for q in chunks:
            if wantreplicate:
                dat = np.c_[np.tile(mtemp[0, q], (3, 0)),
                            mtemp[:, q],
                            np.tile(mtemp[-1, q], (3, 1))]

                temp[q] = pchip(timenew, dat, extrapolate=True)(timeorig)
            else:
                temp[q] = pchip(timenew, mtemp[:, chunks[q]],
                                extrapolate=True)(timeorig)

        pnew.append(temp)

        # prepare output
        msize[dim] = numsamples
        m[p] = reshape2D_undo(m[p], dim, msize)

    return m


def ang2complex(m):
    """[summary]

    Args:
        <m> is a matrix of angles in radians

    Returns:
        [f][complex]: <f> as the corresponding unit-magnitude complex numbers.
        [x][coordinates]: <x> as the x-coordinates of the complex numbers.
        [y][coordinates]: <y> as the y-coordinates of the complex numbers.

    Example:
        f,x,y = ang2complex(.2)
    """

    x = np.cos(m)
    y = np.sin(m)
    f = x + 1j*y

    return f, x, y


def choose(flag, yes, no):
    """
    function f = choose(flag,yes,no)

    <flag> is a truth value (0 or 1)
    <yes> is something
    <no> is something

    if <flag>, return <yes>.  otherwise, return <no>.
    """
    if flag:
        f = yes
    else:
        f = no
    return f


def isrowvector(m):
    """
    f = isrowvector(m)

    function f = isrowvector(m)

    <m> is a matrix

    return whether <m> is 1 x n where n >= 0.
    specifically:
    f = isvector(m) & size(m,1)==1;

    example:
    isrowvector([[1,2]])
    isrowvector([[1]])
    isrowvector(np.zeros(2))
    not isrowvector([])
    """

    import numpy as np

    if not isinstance(m, np.ndarray):
        m = np.asarray(m)
    f = m.shape[0] == 1
    return f


def reshape2D(m, dim):
    """
    shift dimension <dim> to the beginning,
    then reshape to be a 2D matrix.
    see also reshape2D_undo

    Args:
        m ([type]): <m> is a matrix
        dim ([type]): <dim> is a dimension of <m>

    Returns:
        <f> a reshaped 2D matrix

    Example:
        a = np.random.randn(3,4,5)
        b = reshape2D(a,1)
        assert(b.shape == (4, 15))
    """

    # permute and then squish into a 2D matrix
    f = np.moveaxis(m, dim, 0).reshape((m.shape[dim], -1), order='F')

    return f


def reshape2D_undo(f, dim, msize):
    """

    undo operation performed by reshape2D.

    Args:
        f ([type]): <f> has the same dimensions as the output of reshape2D
        dim ([type]): <dim> was the dimension of <m> that was used in reshape2D
        msize ([type]): <msize> was the size of <m>

    Returns:
        <f> but with the same dimensions as passed to reshape2D.

    Example:
        a = np.random.randn(3,4,5)
        b = reshape2D(a,1)
        assert(b.shape == (4, 15))
        c = reshape2D_undo(b,1,a.shape)
        assert(c.shape== (3, 4, 5))
        assert(np.all(a==c))

    """
    # figure out the permutation order that was used in reshape2D
    msizetmp = list(msize)
    msizetmp.remove(msize[dim])
    msizetmp.insert(0, msize[dim])

    # unsquish and the permute back to the original order
    m = np.moveaxis(f.reshape(msizetmp, order='F'), 0, dim)

    return m


def chunking(vect, num, chunknum=None):
    """chunking(v,num)
    Input:
        <v> is a array
        <num> is desired length of a chunk
        <chunknum> is chunk number desired (here we use a 1-based 
              indexing, i.e. you may want the frist chunk, or the second
              chunk, but not the zeroth chunk)
    Returns:
        [numpy array object]:

        return a numpy array object of chunks.  the last vector
        may have fewer than <num> elements.

        also return the beginning and ending indices associated with
        this chunk in <xbegin> and <xend>.

    Examples:

        a = np.empty((2,), dtype=np.object)
        a[0] = [1, 2, 3]
        a[1] = [4, 5]
        assert(np.all(chunking(list(np.arange(5)+1),3)==a))

        assert(chunking([4, 2, 3], 2, 2)==[3])

    """
    if chunknum is None:
        nchunk = int(np.ceil(len(vect)/num))
        f = np.empty((nchunk, ), dtype=np.object)
        for point in range(nchunk):
            f[point] = vect[point*num:np.min((len(vect), int((point+1)*num)))]

        return f
    else:
        nchunk = int(np.ceil(len(vect)/num))
        f = np.empty((nchunk, ), dtype=np.object)
        xbegin = (chunknum-1)*num+1
        xend = np.min((len(vect), chunknum*num))
        for point in range(nchunk):
            f[point] = vect[point*num:np.min((len(vect), int((point+1)*num)))]

        return f[num-1]
