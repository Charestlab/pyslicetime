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
        plt.plot(x0, y0, 'r.-', lw=1)
        xtimes = np.arange(0,.23*len(y1), .23)
        plt.plot(xtimes, y1, 'go', fillstyle='none')
        plt.autoscale(enable=True, axis='both', tight=True)


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

    """ 
    this is how the matlab function is called:

    % episliceorder{1} = [1 14 27 12 25 10 23 8 21 6 19 4 17 2 15 28 13 26 11 24 9 22 7 20 5 18 3 16 1 14 27 12 25 10 23 8 21 6 19 4 17 2 15 28 13 26 11 24 9 22 7 20 5 18 3 16 1 14 27 12 25 10 23 8 21 6 19 4 17 2 15 28 13 26 11 24 9 22 7 20 5 18 3 16];
    % episliceorder{2} = repmat(episliceorder{2},[1 length(epis)]); %this is the desired new resampling tr
    % episliceorder{3} = offset (set to 0 by default in preprocessfmri.m)
    ...
    
    dim_ts = 4 % dims are x * y * 1 * time
    numsamples=[]
    for p =1:length(epis) % one run at a time??
        tr_new = episliceorder{2}(p);
        tr_old = epitr(p);
        for q=1:size(epis{p},3)  % process each slice separately
            this_slice_xyts = single(epis{p}(:,:,q,:));
            this_slice_order= episliceorder{1}(q);
            max_slice = max(episliceorder{1});
            fakeout = -(((1-this_slice_order)/max_slice) * tr_old) - episliceorder{3}(p)

            temp0 = tseriesinterp(this_slice_xyts,tr_old,tr_new,dim_ts,numsamples, ...
                                    fakeout, ...
                                    1,'pchip')

    """

    # internal constants
    numchunks = 20

    # check if m is a vector and make it a stack
    if len(m.shape) == 1:
        m = np.asarray([m])

    # input
    if dim is None:
        dim = len(m.shape)-1

    if numsamples is None:
        numsamples = []

    # do it
    pnew = []
    for p in m:

        # prep 2D
        msize = p.shape
        if len(msize) > 1:
            p = reshape2D(p, dim)
        else:
            p = p[np.newaxis].T

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

            timeorig = \
                [0.0 + x*(trorig*p.shape[0])/len(p) for x in range(len(p))]

            # timeorig = np.linspace(0, trorig*p.shape[0], p.shape[0])
        #  timenew  = linspacefixeddiff(0,trnew,numsamples) - fakeout;

        timenew = [0.0 + x*(trnew*numsamples) /
                   numsamples for x in range(int(numsamples))]

        # do in chunks
        chunks = chunking(
            list(range(p.shape[1])), int(np.ceil(p.shape[1]/numchunks)))
        temp = []
        mtemp = p

        for q in chunks:
            if wantreplicate:
                dat = np.c_[np.tile(mtemp[0, q], (3, 0)),
                            mtemp[:, q],
                            np.tile(mtemp[-1, q], (3, 1))]

                temp[q] = pchip(timeorig, dat, extrapolate=True)(timenew)
            else:

                temp.append(pchip(timeorig, mtemp[:, q],
                                  extrapolate=True)(timenew))
                """
                The code is pretty much working up to here.
                not sure how to deal with the reshape2D stuff tho.
                    
                """

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
    """ chunking
    Input:
        <vect> is a array
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

        assert(chunking([4, 2, 3], 2, 2)==([3], 3, 3))

    """
    if chunknum is None:
        nchunk = int(np.ceil(len(vect)/num))
        f = []
        for point in range(nchunk):
            f.append(vect[point*num:np.min((len(vect), int((point+1)*num)))])

        return np.asarray(f)
    else:
        f = chunking(vect, num)
        # double check that these behave like in matlab (xbegin)
        xbegin = (chunknum-1)*num+1
        # double check that these behave like in matlab (xend)
        xend = np.min((len(vect), chunknum*num))

        return np.asarray(f[num-1]), xbegin, xend
