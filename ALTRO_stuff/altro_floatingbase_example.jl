using RigidBodyDynamics
using StaticArrays
using Attitude
using LinearAlgebra
using TrajectoryOptimization
using Altro
using RobotDynamics
using MATLAB
const TO = TrajectoryOptimization

struct FloatingSat{T} <: TO.AbstractModel
    mass::T
end
# dynamics stuff
function RobotDynamics.dynamics(model::FloatingSat, x::AbstractVector{T}, u) where T
    """Solver state and RBD state differ, note below

         # solver state
         [configuration]
         p = x[1:3] <: q
         r = x[4:6] <: q
         θ = x[7:9] <: q

         [velocity]
         ω = x[10:12]
         vel = x[13:15]
         θ̇ = x[16:18]

         # RBD state
         [configuration]
         quat = x[1:4] <: q
         r = x[5:7] <: q
         θ = x[8:10] <: q

         [velocity]
         ω = x[11:13]
         vel = x[14:16]
         θ̇ = x[17:19]

    """

    # solver state
    # configuration stuff
    p = x[1:3]
    r = x[4:6]
    θ = x[7:9]

    # velocity stuff
    ω = x[10:12]
    vel = x[13:15]
    θ̇ = x[16:18]
    # θ̇ = @SVector [x[16],x[17],x[18]]

    # now we convert it to a state for RBD
    state = statecache[T]
    copyto!(state,[q_from_p(p);x[4:end]])

    # get the dynamics for v (this state is the same for both)
    M = Array(mass_matrix(state))
    if hasnan(M)
        return NaN*x
    else
        # dynamics
        v̇ = (M)\(-dynamics_bias(state) + u)

        # kinematics
        q̇ = [pdot_from_w(p,ω);vel;θ̇]

        return [q̇;v̇]
    end
end

urdf = "/Users/kevintracy/devel/SatelliteServicing/ALTRO_stuff/simple_sat.xml"

mechanism = parse_urdf(urdf,gravity = [0; 0; 0],floating = true)
const statecache = StateCache(mechanism)

Base.size(::FloatingSat) = 18,9

# Model and discretization
model = FloatingSat(1.0)
n,m = size(model)
tf = 10.0  # sec
N = 75    # number of knot points

# Objective
x0 = SVector{18}(zeros(18))  # initial state
xf = SVector{18}([.414;.3;.1;5;2;3;[pi;deg2rad(45);-deg2rad(90)];zeros(9)])

Q = Diagonal([100*ones(3);10*ones(3);10*ones(3);.1*ones(3);.1*ones(3);.1*ones(3)])
Qf = 100*Q
R = .1*Diagonal([ones(3);ones(3);50*ones(3)])
obj = LQRObjective(Q, R, Qf, xf, N)

# Constraints
cons = ConstraintList(n,m,N)
# add_constraint!(cons, GoalConstraint(xf), N)
add_constraint!(cons, BoundConstraint(n,m, u_min=-5.0, u_max=5.0), 1:N-1)

# Create and solve problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)

opts = SolverOptions(verbose =2,
                     static_bp = false)
solver = ALTROSolver(prob,opts)
cost(solver)           # initial cost
solve!(solver)         # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories
X = states(solver)
U = controls(solver)

Xm = mat_from_vec(X)

Um = mat_from_vec(U)

mat"
figure
title('X')
hold on
plot($Xm')
hold off
"

mat"
figure
hold on
title('U')
plot($Um')
hold off
"
