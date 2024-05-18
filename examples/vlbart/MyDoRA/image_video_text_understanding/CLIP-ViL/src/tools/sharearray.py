# Copyright 2017 Brendan Shillingford
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import os.path
import re
import time
import sys

import numpy as np

__all__ = ['cache', 'decorator', 'valid_id', 'TimeoutException']


if sys.version_info[0] == 2:
     FileExistsError = OSError


class TimeoutException(Exception):
    pass


_ID_REGEX = re.compile(r"^[A-Za-z0-9=+._-]+$")


def valid_id(id):
    if _ID_REGEX.match(id):
        return True
    return False


def _memmapped_view(filename):
    return np.lib.format.open_memmap(filename, mode='r')


def _build_path(id, prefix, shm_path):
    fn = os.path.join(shm_path, prefix + id + '.npy')
    fn_lock = fn + '.lock'
    return fn, fn_lock


def free(id, shm_path='/dev/shm', prefix='sharearray_'):
    fn, fn_lock = _build_path(id, prefix=prefix, shm_path=shm_path)
    fn_exists = os.path.exists(fn)
    fn_lock_exists = os.path.exists(fn_lock)

    if fn_lock_exists:
        import warnings
        warnings.warn("lock still exists")
        os.unlink(fn_lock)

    if fn_exists:
        os.unlink(fn)


def cache(id, array_or_callback,
          shm_path='/dev/shm',
          prefix='sharearray_',
          timeout=-1,
          verbose=True,
          log_func=None):
    """
    Stores a `numpy` `ndarray` into shared memory, caching subsequent requests
    (globally, across all processes) to the function so they point to the same
    memory.

    By default, does this be creating a file at `/dev/shm/shareddataset_<id>`.
    If:

    1. The file is not created yet, saves `array_or_callback` to the path
       listed above (see NOTE 1). Then, returns a read-only memmapped view to
       this numpy array.

    2. The file is already created. We return a read-only memmapped view of it.

    Args:
        id (str): identifier for shared array, global across system.
            Must match `[A-Za-z0-9=+._-]+`. You may want to include your
            program's name to prevent name collisions.
        array_or_callback: either a `numpy.ndarray` containing value types, or
            a callback function taking no arguments that returns one.
        shm_path (str, optional): path to the Linux shared memory
            tmpfs mountpoint. In almost all kernel builds, one lives at
            `/dev/shm` with size defaulting to half the RAM; sometimes it's a
            symlink to `/run/shm`.
        timeout (int, optional): number of seconds to wait before timing out
            waiting for lock to be released, when file is already being created.
            If -1, waits indefinitely.
        prefix (str, optional): prefix added to files in `shm_path`.
        verbose (bool): if True, prints useful information (2-3 lines).
        log_func (callable): if verbose is True, this is used if specified.
            Else just uses `print`.

    Returns:
        A `numpy.ndarray` read-only view into the shared memory, whether it
        was newly created or previously created.

    Raises:
        ValueError: `id` is not a valid identifier (must match
            `[A-Za-z0-9=+._-]+`), or `array_or_callback` is not a callback
            or returns
        TimeoutException: if `timeout` is positive, the lock file exists, and
            we have waited at least `timeout` seconds yet the lock still exists.

    Notes:

        NOTE 1: For concurrency safety, this function creates a lock file at
        `/dev/shm/shareddataset_<id>.lock` when initially writing the file
        (lock file is empty, doesn't contain PID). File creating is hence
        checked using the lock, rather than the file's existence itself. We
        don't use the standard create-rename method of ensuring atomicity,
        since `array_or_callback` may be expensive to call or large to write.

        NOTE 2: `id`s are currently global to the system. Include your program's
        name to prevent name collisions.

        NOTE 3: memmapped views are created using
        `numpy.lib.format.open_memmap`.

        NOTE 4: If the array is very large, to save memory, you may want to
        immediately remove all references to the original array, then do a full
        garbage collection (`import gc; gc.collect()`).

    Examples:

        An expensive operation (e.g. data preprocessing) that is the same
        across all running instances of this program:

            x_times_y = cache("myprog_x_times_y", lambda: np.dot(x, y))

        A large (large enough to warrant concern, but small enough to fit in
        RAM once) training set that we only want one instance of across
        many training jobs:

            def load_training_set():
                # load and/or preprocess training_set once here
                return training_set
            training_set = cache("myprog_training_set", load_training_set)

        Only passing a callback to array_or_callback makes sense here,
        of course.
    """
    if not valid_id(id):
        raise ValueError('invalid id: ' + id)

    '''if not (hasattr(array_or_callback, '__call__')
            or isinstance(array_or_callback, np.ndarray)):
        raise ValueError(
            'array_or_callback should be ndarray or zero-argument callable')'''

    if verbose and log_func:
        print_ = log_func
    elif verbose:
        def print_(s):
            print(s)
    else:
        def print_(s):
            pass

    fn, fn_lock = _build_path(id, prefix=prefix, shm_path=shm_path)
    fd_lock = -1
    try:
        fd_lock = os.open(fn_lock, os.O_CREAT | os.O_EXCL)
        if fd_lock < 0:
            raise OSError("Lock open failure (bug?)", fn_lock, fd_lock)

    except FileExistsError:
        if timeout < 0:
            #print_(("'{}' is being created by another process. "
            #        "Waiting indefinitely... (timeout < 0)").format(id))

            while os.path.exists(fn_lock):
                time.sleep(1)

        else:
            #print_(("'{}' is being created by another process. "
            #        "Waiting up to {} seconds...").format(id, timeout))

            for _ in range(timeout):
                time.sleep(1)
                if not os.path.exists(fn_lock):
                    break
            else:
                raise TimeoutException(
                    "timed out waiting for %s to unlock (be created)" % id)
    else:
        if not os.path.exists(fn):
            print_("'%s' doesn't exist yet. Locking and creating..." % id)
            if hasattr(array_or_callback, '__call__'):
                array = array_or_callback()
            else:
                array = array_or_callback
            '''if isinstance(array_or_callback, np.ndarray):
                array = array_or_callback
            else:
                array = array_or_callback()
                if not isinstance(array, np.ndarray):
                    raise ValueError(
                        'callback did not return a numpy.ndarray, returned:',
                        type(array))'''

            np.save(fn, array, allow_pickle=False)
            print_("'%s': written." % id)

    finally:
        if fd_lock > 0:
            os.close(fd_lock)
            os.unlink(fn_lock)

    print_("'%s': returning memmapped view." % id)
    return _memmapped_view(fn)

def cache_with_delete_previous(id, array_or_callback,
          shm_path='/dev/shm',
          prefix='sharearray_',
          timeout=-1,
          verbose=True,
          log_func=None,
          delete=None,
          wait=0):
    """
    Stores a `numpy` `ndarray` into shared memory, caching subsequent requests
    (globally, across all processes) to the function so they point to the same
    memory.

    By default, does this be creating a file at `/dev/shm/shareddataset_<id>`.
    If:

    1. The file is not created yet, saves `array_or_callback` to the path
       listed above (see NOTE 1). Then, returns a read-only memmapped view to
       this numpy array.

    2. The file is already created. We return a read-only memmapped view of it.

    Args:
        id (str): identifier for shared array, global across system.
            Must match `[A-Za-z0-9=+._-]+`. You may want to include your
            program's name to prevent name collisions.
        array_or_callback: either a `numpy.ndarray` containing value types, or
            a callback function taking no arguments that returns one.
        shm_path (str, optional): path to the Linux shared memory
            tmpfs mountpoint. In almost all kernel builds, one lives at
            `/dev/shm` with size defaulting to half the RAM; sometimes it's a
            symlink to `/run/shm`.
        timeout (int, optional): number of seconds to wait before timing out
            waiting for lock to be released, when file is already being created.
            If -1, waits indefinitely.
        prefix (str, optional): prefix added to files in `shm_path`.
        verbose (bool): if True, prints useful information (2-3 lines).
        log_func (callable): if verbose is True, this is used if specified.
            Else just uses `print`.

    Returns:
        A `numpy.ndarray` read-only view into the shared memory, whether it
        was newly created or previously created.

    Raises:
        ValueError: `id` is not a valid identifier (must match
            `[A-Za-z0-9=+._-]+`), or `array_or_callback` is not a callback
            or returns
        TimeoutException: if `timeout` is positive, the lock file exists, and
            we have waited at least `timeout` seconds yet the lock still exists.

    Notes:

        NOTE 1: For concurrency safety, this function creates a lock file at
        `/dev/shm/shareddataset_<id>.lock` when initially writing the file
        (lock file is empty, doesn't contain PID). File creating is hence
        checked using the lock, rather than the file's existence itself. We
        don't use the standard create-rename method of ensuring atomicity,
        since `array_or_callback` may be expensive to call or large to write.

        NOTE 2: `id`s are currently global to the system. Include your program's
        name to prevent name collisions.

        NOTE 3: memmapped views are created using
        `numpy.lib.format.open_memmap`.

        NOTE 4: If the array is very large, to save memory, you may want to
        immediately remove all references to the original array, then do a full
        garbage collection (`import gc; gc.collect()`).

    Examples:

        An expensive operation (e.g. data preprocessing) that is the same
        across all running instances of this program:

            x_times_y = cache("myprog_x_times_y", lambda: np.dot(x, y))

        A large (large enough to warrant concern, but small enough to fit in
        RAM once) training set that we only want one instance of across
        many training jobs:

            def load_training_set():
                # load and/or preprocess training_set once here
                return training_set
            training_set = cache("myprog_training_set", load_training_set)

        Only passing a callback to array_or_callback makes sense here,
        of course.
    """
    if not valid_id(id):
        raise ValueError('invalid id: ' + id)

    '''if not (hasattr(array_or_callback, '__call__')
            or isinstance(array_or_callback, np.ndarray)):
        raise ValueError(
            'array_or_callback should be ndarray or zero-argument callable')'''

    if verbose and log_func:
        print_ = log_func
    elif verbose:
        def print_(s):
            print(s)
    else:
        def print_(s):
            pass

    fn, fn_lock = _build_path(id, prefix=prefix, shm_path=shm_path)
    fd_lock = -1
    try:
        fd_lock = os.open(fn_lock, os.O_CREAT | os.O_EXCL)
        if fd_lock < 0:
            raise OSError("Lock open failure (bug?)", fn_lock, fd_lock)

    except FileExistsError:
        if timeout < 0:
            #print_(("'{}' is being created by another process. "
            #        "Waiting indefinitely... (timeout < 0)").format(id))

            while os.path.exists(fn_lock):
                time.sleep(1)

        else:
            #print_(("'{}' is being created by another process. "
            #        "Waiting up to {} seconds...").format(id, timeout))

            for _ in range(timeout):
                time.sleep(1)
                if not os.path.exists(fn_lock):
                    break
            else:
                raise TimeoutException(
                    "timed out waiting for %s to unlock (be created)" % id)
    else:
        if not os.path.exists(fn):
            print("Sleep a bit for other workers to finish whatever they are doing")
            for _ in range(wait):
                time.sleep(1)
            if delete is not None:
                print("Deleting {} on the way there".format(delete))
                for i in delete:
                    if os.path.exists(i):
                        os.remove(i)

            print_("'%s' doesn't exist yet. Locking and creating..." % id)
            if hasattr(array_or_callback, '__call__'):
                array = array_or_callback()
            else:
                array = array_or_callback
            
            '''if isinstance(array_or_callback, np.ndarray):
                array = array_or_callback
            else:
                array = array_or_callback()
                if not isinstance(array, np.ndarray):
                    raise ValueError(
                        'callback did not return a numpy.ndarray, returned:',
                        type(array))'''

            np.save(fn, array, allow_pickle=False)
            print_("'%s': written." % id)

    finally:
        if fd_lock > 0:
            os.close(fd_lock)
            os.unlink(fn_lock)

    #print_("'%s': returning memmapped view." % id)
    return _memmapped_view(fn)


def decorator(id, **kwargs):
    """
    Decorator version of `cache`, analogous to a memoization decorator.

    Besides `array_or_callback` which isn't needed, arguments are identical to
    those of `cache`, see there for docs. They must be passed as keyword args
    except for `id`.

    Note that `id` can't depend on the arguments to the decorated function.
    For that, use `cache` directly.

    Example:
        Alternative to callback syntax above.

            @decorator("my_large_array")
            def foo():
                # ...do some expensive computation to generate arr...
                return arr

            arr = foo()  # first call, in shared memory arr global to system
            arr2 = foo() # here or another script, returns read-only view
    """
    if not valid_id(id):
        raise ValueError('invalid id: ' + id)

    def decorate(f):
        @functools.wraps(f)
        def wrapped():
            return cache(id, f, **kwargs)

        return wrapped

    return decorate