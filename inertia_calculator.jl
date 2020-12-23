using LinearAlgebra


function inertcalc()
mass = 1

L = .1
W = .1
H = 2

Ixx = (mass/12)*(W^2 + H^2)
Iyy = (mass/12)*(L^2 + H^2)
Izz = (mass/12)*(L^2 + W^2)

@info "new shape"
@show Ixx
@show Iyy
@show Izz
end

inertcalc()
