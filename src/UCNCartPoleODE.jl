module UCNCartPoleODE

using StaticArrays
using Infiltrator

const nx = 5    # number of states
const nd = 4    # degrees of freedom
const nu = 1    # number of inputs

# x = [s cos(θ/2) sin(θ/2) v ω]

struct Model
    g    # ms⁻² - gravity
    m_c  # kg - mass of the cart
    m_p  # kg - mass of the point-mass 
    l    # m - length of the pole
end

# Unit complex numbers q = cos(θ/2) + i * sin(θ/2)

"""complex conjugate"""
conjugate(q) = @SVector [q[1], -q[2]]

"""Composition of rotations q1 and q2"""
multiply(q1, q2) = @SVector [
    q1[1] * q2[1] - q1[2] * q2[2],
    q1[1] * q2[2] + q1[2] * q2[1]
]

"""2D rotation matrix"""
rot(q) = @SMatrix [
    qcos(q) -qsin(q)
    qsin(q) qcos(q)
]

"""conversion to sines and cosines"""
qsin(q) = 2 * q[2] * q[1] # 2 * sin(θ/2) * cos(θ/2)
qcos(q) = q[1]^2 - q[2]^2 # cos^2(θ/2) - sin^2(θ/2)

# CartPole

## dynamics
mass_matrix(m, x) = @SMatrix [
    m.m_c+m.m_p m.m_p*m.l*qcos(x[2:3])
    m.m_p*m.l*qcos(x[2:3]) m.m_p*m.l^2
]

torque_vector(m, x, u) = @SVector [
    m.m_p * m.l * qsin(x[2:3]) * x[5]^2 + u[1],
    -m.g * m.m_p * m.l * qsin(x[2:3])
]

dxdz(x, ż) = @SVector [
    ż[1], -0.5 * x[3] * ż[2], 0.5 * x[2] * ż[2], ż[3], ż[4]
]

function f(m, x, u)
    M = mass_matrix(m, x)
    τ = torque_vector(m, x, u)

    return Vector(dxdz(x, vcat(x[4:5], M \ τ)))
end

function f!(m, ẋ, x, u)
    ẋ .= f(m, x, u)
    return nothing
end

## jacobian to the tangent
jacobian(x) = @SMatrix [
    1 0 0 0
    0 -x[3] 0 0
    0 x[2] 0 0
    0 0 1 0
    0 0 0 1
]

## state difference
function state_difference(x, x0)
    dq = multiply(conjugate(x0[2:3]), x[2:3])

    ds = x[1] - x0[1]
    dθ = dq[2]
    dv = x[4] - x0[4]
    dω = x[5] - x0[5]

    return vcat(ds, dθ, dv, dω)
end

end # module UCNCartPoleODE
