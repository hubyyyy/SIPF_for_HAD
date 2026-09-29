# Numerical Algorithm

## Coordinates and Fourier normalization

The physical box is

```text
Omega = [-L/2, L/2)^3.
```

The implementation stores `y = (x + L/2) mod L` and expands a field as

```text
u(y) = sum_j u_hat[j] exp(i 2 pi j dot y / L).
```

For a uniform `N^3` grid, the code uses

```text
u_hat = fftn(u) / N^3
u     = real(ifftn(u_hat) * N^3).
```

Using shifted particle positions in the empirical Fourier coefficient

```text
rho_hat[j] = (M0/P) sum_p exp(-i 2 pi j dot y_p / L)
```

is algebraically equivalent to applying the alternating phase factor generated
by shifting the centered physical box to the FFT box. No second phase factor
should be applied.

For an even number of modes, the continuous positive Nyquist partner is absent.
The implementation therefore zeros the Nyquist planes and enforces conjugate
symmetry. This makes off-grid Fourier evaluations at particle positions real.

## One time step

Given `(X_n, f_hat_n, m_hat_n)`, the code applies:

```text
X_(n+1)^p = X_n^p
          + gamma dt grad(f_n)(X_n^p)
          + sqrt(2 d1 dt) Normal(0, I),
```

followed by periodic wrapping.

The ECM coefficients use explicit Euler:

```text
f_hat_(n+1) = f_hat_n - eta dt P_H(m_n f_n).
```

`P_H(m_n f_n)` is computed by packing the retained modes into a grid with at
least twice as many modes in every direction, multiplying in physical space,
and extracting the original modes. This is the FFT realization of the valid
mode interactions stated in the paper.

Finally, direct empirical coefficients from `X_(n+1)` enter the implicit MDE
update:

```text
m_hat_(n+1)[j] =
    (m_hat_n[j] + alpha dt rho_hat_(n+1)[j])
    / (1 + dt (d2 |2 pi j/L|^2 + beta)).
```

This coefficient formula is the direct Fourier form of the modified Helmholtz
step in Section 3.

## Initial data

The radial configuration uses

```text
rho_0 = exp(-r^2/epsilon), r <= 0.1,
f_0   = 1 - 0.5 rho_0,
m_0   = 0.5 rho_0,
```

with the paper values

```text
d1=d2=0.001, gamma=0.005, eta=10,
alpha=0.1, beta=0, epsilon=0.0025.
```

The two-cluster configuration adds identical profiles centered at
`(0.08,0.08,0.08)` and `(-0.08,-0.08,-0.08)`.

Particles are sampled from the normalized truncated Gaussian radial density.
Their common weight is `M0/P`, where `M0` is evaluated analytically.
