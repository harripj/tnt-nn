import numpy as np
import numba


@numba.njit
def mdot(a, b):
    """
    Emulate Matlab dot function.
    """
    return (a.conj() * b).sum(axis=0)


# @numba.njit
def is_pos_def(arr):
    """
    Test whether an array is positive definite.

    Returns
    -------
    res: bool
        True if positive_definite.
    """
    return np.all(np.linalg.eigvals(arr) > 0)


@numba.njit
def H(arr):
    """
    Return Hermitian of complex array.

    Parameters
    ----------
    arr: ndarray

    Returns
    -------
    arr.H: ndarray
        The Hermitian of the array.
    """
    return arr.conj().T


def nnls_tnt(A, b, lam=0.0, rel_tol=0.0, red_c=0.2, exp_c=1.2):
    """
    Emulate nnls_tnt from tnt.m.

    TNT-NNLS as described in [1]. Translated from Matlab code.

    Parameters
    ----------
    A: (M, N) ndarray
    b: (N,) ndarray

    Returns
    -------
    x: (M,) ndarray
        Least squares result.
    ...
    """
    m, n = A.shape
    AA = np.dot(H(A), A)
    dtype = AA.dtype

    # define small epsilon value, related to the size of A
    # force same dtypes

    epsilon = dtype.type(10 * np.finfo(A.dtype).eps) * np.linalg.norm(AA, 1)
    AA = AA + epsilon * np.eye(n, dtype=dtype)

    # emulate lsq_solver in tnt.m
    x = np.zeros(n, dtype=dtype)
    free_set = np.arange(n)[::-1]
    binding_set = np.zeros(
        0, dtype=int
    )  # in Matlab code this var is initialized then set to empty list...
    insertion_set = np.zeros(n, dtype=int)
    residual = np.zeros(n, dtype=dtype)
    gradient = np.zeros(n, dtype=dtype)

    score, x, residual, free_set, binding_set, AA, epsilon, dels, lps = lsq_solve(
        A, b, lam, AA, epsilon, free_set, binding_set, n
    )

    # ===============================================================
    # Outer Loop.
    # ===============================================================
    OuterLoop = 0
    TotalInnerLoops = 0
    insertions = n
    while True:
        OuterLoop += 1
        # ===============================================================
        # Save this solution.
        # ===============================================================
        best_score = score
        best_x = x
        best_free_set = free_set
        best_binding_set = binding_set
        best_insertions = insertions
        max_insertions = int(exp_c // best_insertions)

        # ===============================================================
        # Compute the gradient of the "Normal Equations".
        # ===============================================================
        gradient = np.dot(H(A), residual)

        # ===============================================================
        # Check the gradient components.
        # ===============================================================
        insertions = 0
        tempg = gradient[binding_set]
        insertion_set = np.nonzero(tempg)[0]
        insertions += len(insertion_set)

        # ===============================================================
        # Are we done ?
        # ===============================================================
        if not insertions:
            # there were no changes that were feasible- we are done.
            status = 0  # success
            break

        # ===============================================================
        # Sort the possible insertions by their gradients to find the
        # most attractive variables to insert.
        # ===============================================================
        grad_score = gradient[binding_set[insertion_set]]
        set_index = np.argsort(grad_score)[::-1]
        insertion_set = insertion_set[set_index]

        # ===============================================================
        # Inner Loop.
        # ===============================================================
        InnerLoop = 0
        while True:
            InnerLoop += 1
            TotalInnerLoops += 1

            # ==============================================================
            # Adjust the number of insertions.
            # ==============================================================
            insertions = int(red_c * insertions)
            if not insertions:
                insertions = 1
            elif insertions > max_insertions:
                insertions = max_insertions
            insertion_set = insertion_set[:insertions]

            # ==============================================================
            # Move variables from "binding" to "free".
            # ==============================================================
            free_set = np.concatenate([free_set, binding_set[insertion_set]])
            binding_set = np.delete(binding_set, insertion_set)

            # ===============================================================
            # Compute a feasible solution using the unconstrained
            # least-squares solver of your choice.
            # ===============================================================
            (
                score,
                x,
                residual,
                free_set,
                binding_set,
                AA,
                epsilon,
                dels,
                lps0,
            ) = lsq_solve(A, b, lam, AA, epsilon, free_set, binding_set, insertions)

            # ===============================================================
            # Check for new best solution.
            # ===============================================================
            if score < best_score * (1.0 - rel_tol):
                break

            # ===============================================================
            # Restore the best solution.
            # ===============================================================
            score = best_score
            x = best_x
            free_set = best_free_set
            binding_set = best_binding_set
            max_insertions = int(exp_c * best_insertions)

            # ===============================================================
            # Are we done ?
            # ===============================================================
            if insertions == 1:
                # the best feasible change did not improve the score- we are done
                status = 0  # success
                return x, AA, status, OuterLoop, TotalInnerLoops

    return x, AA, status, OuterLoop, TotalInnerLoops


def lsq_solve(A, b, lam, AA, epsilon, free_set, binding_set, deletions_per_loop):
    """
    Emulate the lsq_solve fn from tnt.m.
    """
    free_set = np.sort(free_set)[::-1]
    binding_set = np.sort(binding_set)[::-1]

    # BB should be a square array
    B = A[:, free_set]
    BB = AA[free_set[:, np.newaxis], free_set]

    if lam > 0:
        for i in range(free_set.size):
            B[i, i] = B[i, i] + lam
            BB[i, i] = BB[i, i] + lam * lam

    if is_pos_def(BB):
        R = H(np.linalg.cholesky(BB))
    else:
        while True:
            epsilon *= 10
            AA += np.eye(AA.shape[0], AA.shape[1]) * epsilon
            BB = AA[free_set[:, np.newaxis], free_set]
            if lam > 0:
                for i in range(free_set.size):
                    BB[i, i] = BB[i, i] + lam * lam
            if is_pos_def(BB):
                R = H(np.linalg.cholesky(BB))
                break
            else:
                continue

    # loop until solution is feasible
    dels = 0
    loops = 0
    lsq_loops = 0
    del_hist = np.zeros(0, dtype=int)

    while True:
        loops += 1
        # ------------------------------------------------------------
        # Use PCGNR to find the unconstrained optimum in
        # the "free" variables.
        # ------------------------------------------------------------
        reduced_x, k = pcgnr(B, b, R)

        if k > lsq_loops:
            lsq_loops = k

        # ------------------------------------------------------------
        # Get a list of variables that must be deleted.
        # ------------------------------------------------------------
        deletion_set = np.nonzero(reduced_x <= 0)[0]

        # ------------------------------------------------------------
        # If the current solution is feasible then quit.
        # ------------------------------------------------------------
        if not len(deletion_set):
            break

        # ------------------------------------------------------------
        # Sort the possible deletions by their reduced_x values to
        # find the worst violators.
        # ------------------------------------------------------------
        x_score = reduced_x[deletion_set]
        set_index = np.argsort(x_score)
        deletion_set = deletion_set[set_index]

        # ------------------------------------------------------------
        # Limit the number of deletions per loop.
        # ------------------------------------------------------------
        if deletion_set.size > deletions_per_loop:
            deletion_set = deletion_set[:deletions_per_loop]

        deletion_set = np.sort(deletion_set)[::-1]
        del_hist = set(del_hist).union(set(deletion_set))
        # del_hist = np.union1d(del_hist, deletion_set)
        dels = dels + deletion_set.size

        # ------------------------------------------------------------
        # Move the variables from "free" to "binding".
        # ------------------------------------------------------------
        binding_set = np.concatenate([binding_set, free_set[deletion_set]])
        free_set = np.delete(free_set, deletion_set)

        # ------------------------------------------------------------
        # Reduce A to B.
        # ------------------------------------------------------------
        # B is a matrix that has all of the rows of A, but its
        # columns are a subset of the columns of A. The free_set
        # provides a map from the columns of B to the columns of A.
        B = A[:, free_set]

        # ------------------------------------------------------------
        # Reduce AA to BB.
        # ------------------------------------------------------------
        # BB is a symmetric matrix that has a subset of rows and
        # columns of AA. The free_set provides a map from the rows
        # and columns of BB to rows and columns of AA.
        BB = AA[free_set[:, np.newaxis], free_set]

        # ------------------------------------------------------------
        # Adjust with Tikhonov regularization parameter lambda.
        # ------------------------------------------------------------
        if lam > 0:
            for i in range(free_set.size):
                B[i, i] = B[i, i] + lam
                BB[i, i] = BB[i, i] + lam * lam

        # ------------------------------------------------------------
        # Compute R, the Cholesky factor.
        # ------------------------------------------------------------
        R = cholesky_delete(R, BB, deletion_set)

    # ------------------------------------------------------------
    # Unscramble the column indices to get the full (unreduced) x.
    # ------------------------------------------------------------
    x = np.zeros(A.shape[1])
    x[free_set] = reduced_x

    # ------------------------------------------------------------
    # Compute the full (unreduced) residual.
    # ------------------------------------------------------------
    residual = b - np.dot(A, x)

    # ------------------------------------------------------------
    # Compute the norm of the residual.
    # ------------------------------------------------------------
    score = np.sqrt(mdot(residual, residual))

    return score, x, residual, free_set, binding_set, AA, epsilon, dels, loops


@numba.njit
def pcgnr(A, b, R, atol=1e-6):
    """
    Emulate pcgnr from tnt.m.
    """
    m, n = A.shape
    x = np.zeros(n, dtype=A.dtype)
    r = b
    r_hat = np.dot(H(A), r)  # % matrix_x_vector, O(mn)
    y = np.linalg.solve(H(R), r_hat)  # back_substitution, O(n^2)
    z = np.linalg.solve(R, y)  # back_substitution, O(n^2)
    p = z

    gamma = mdot(z, r_hat)
    prev_rr = -1

    for k in range(n):
        w = np.dot(A, p)  # matrix_x_vector, O(mn)
        ww = mdot(w, w)
        if np.abs(ww) <= atol:
            break

        alpha = gamma / ww
        x_prev = x
        x = x + alpha * p
        r = b - np.dot(A, x)  # matrix_x_vector, O(mn)
        r_hat = np.dot(H(A), r)  # matrix_x_vector, O(mn)
        x_prev = x

        rr = mdot(r_hat, r_hat)
        if prev_rr >= 0 and prev_rr <= rr:
            x = x_prev
            break

        prev_rr = rr
        y = np.linalg.solve(H(R), r_hat)  # back_substitution, O(n^2)
        z = np.linalg.solve(R, y)  # back_substitution, O(n^2)
        gamma_new = mdot(z, r_hat)
        beta = gamma_new / gamma
        p = z + beta * p
        gamma = gamma_new
        if np.abs(gamma) <= atol:
            break

    return x, k


# @numba.njit
def cholesky_delete(R, BB, deletion_set):
    """
    Emulate the cholesky_delete fn from tnt.m.
    """

    m, n = R.shape
    num_deletions = len(deletion_set)

    speed_fudge_factor = 0.001
    if num_deletions > (speed_fudge_factor * n):
        # =============================================================
        # Full Cholesky decomposition of BB (on GPUs).
        # =============================================================
        if is_pos_def(BB):
            R = H(np.linalg.cholesky(BB))  # O(n^3/3)
        else:
            # This should never happen because we have already added
            # a sufficiently large "epsilon" to AA to do the
            # nonnegativity tests required to create the deleted_set.
            raise ValueError("This should not happen!")
    else:
        for i in range(num_deletions):
            j = deletion_set[i]

            # =============================================================
            # This function is just a stripped version of Matlab's qrdelete.
            # Stolen from:
            # http://pmtksupport.googlecode.com/svn/trunk/lars/larsen.m
            # =============================================================
            R = np.delete(R, j, axis=1)  # % remove column j
            n = R.shape[1]
            for k in range(j, n):
                # p = range(k, k + 1)
                p = k
                G = givens_qr(R[p, k])  # remove extra element in col
                if k < n:
                    R[p, np.arange(k, n)] = G * R(
                        p, np.arange(k, n)
                    )  # adjust rest of row

            R = R[:-1, :]  # remove zero'ed out row
    return R


# GvL pg. 216 : algo 5.1.3 * see also anderson(2000) via wikipedia for continuity concerns
def zeroing_givens_coeffs(x, z):
    """for the values x,z compute cos th, sin th
    s.t. applying a Givens rotation G(cos th,sin th)
         on 2 rows(or cols) with values x,z will
         maps x --> r and z --> 0"""
    if z == 0.0:  # better:  abs(z) < np.finfo(np.double).eps
        return 1.0, 0.0
    r = np.hypot(x, z)  # C99 hypot is safe for under/overflow
    return x / r, -z / r


# GvL, pg. 216 .... Section 5.1.9
def left_givensT(cs, A, r1, r2):
    """ update A <- G.T.dot(A) ... affects rows r1 and r2 """
    c, s = cs
    givensT = np.array([[c, -s], [s, c]])  # manually transposed
    A[[r1, r2], :] = np.dot(givensT, A[[r1, r2], :])


# A.dot(G) .... affects two cols of A
def right_givens(cs, A, c1, c2):
    """ update A <- A.dot(G) ... affects cols c1 and c2 """
    c, s = cs
    givens = np.array([[c, s], [-s, c]])
    A[:, [c1, c2]] = np.dot(A[:, [c1, c2]], givens)


def givens_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    for c in range(n):
        for r in reversed(range(c + 1, m)):  # m-1, m-2, ... c+2, c+1
            # in this row and the previous row, use zeroing givens to
            # place a zero in the lower row
            coeffs = zeroing_givens_coeffs(A[r - 1, c], A[r, c])
            left_givensT(coeffs, A[:, c:], r - 1, r)
            # left_givensT(coeffs, A[r-1:r+1, c:], 0, 1)
            left_givensT(coeffs, Q[:, c:], r - 1, r)
    return Q