
using DifferentialEquations
function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - u[1]*u[2]
  du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)

sol = solve(prob,Tsit5())

ptr=100
t = collect(range(0,stop=10,length=ptr))



using RecursiveArrayTools
randomized = VectorOfArray([(sol(t[i])) for i in 1:ptr])
data = convert(Array,randomized)
x=Vector{Float64}(undef,ptr)
y=Vector{Float64}(undef,ptr)

for i in range(1,stop=ptr)
  x[i]=data[1,i]
end

for i in range(1,stop=ptr)
  y[i]=data[2,i]
end

using BasisFunctionExpansions

rbf_x = UniformRBFE(t,24, normalize=true) #value 24 and 26 are form lcurve on 400 points
rbf_y = UniformRBFE(t,26, normalize=true)
bfa_x = BasisFunctionApproximation(x,t,rbf_x,1)
bfa_y = BasisFunctionApproximation(y,t,rbf_y,1)


t = collect(range(0,stop=10,length=ptr))

bfa_xp=bfa_x(t)
loss_x=Vector{Float64}(undef,ptr)
for n in range(1,stop=ptr)
 loss_x[n]=x[n]-bfa_xp[n]
end

bfa_yp=bfa_y(t)
loss_y=Vector{Float64}(undef,ptr)
for n in range(1,stop=ptr)
 loss_y[n]=y[n]-bfa_yp[n]
end


new_x = collect(range(0,stop=10,length=ptr))

for i in range(1,stop=ptr)

  if loss_x[i].>1.3
    u=i/10
    o = collect(range(u,stop=u+0.1,length=20))
    append!(new_x,o)
  end
end

new_y = collect(range(0,stop=10,length=ptr))

for i in range(1,stop=ptr)

  if loss_y[i].>1.3
    u=i/10
    o = collect(range(u,stop=u+0.1,length=20))
    append!(new_y,o)
  end
end

h=length(new_x)
randomized_nx = VectorOfArray([(sol(new_x[i])) for i in 1:h])
data_nx = convert(Array,randomized_nx)
x_n=Vector{Float64}(undef,h)

xx=Vector{Float64}(undef,h)
for i in range(1,stop=h)
  xx[i]=data_nx[1,i]
end

j=length(new_y)
randomized_ny = VectorOfArray([(sol(new_y[i])) for i in 1:j])
data_ny = convert(Array,randomized_ny)
y_n=Vector{Float64}(undef,j)
yy=Vector{Float64}(undef,j)
for i in range(1,stop=j)
  yy[i]=data_ny[2,i]
end

rbf_xn = UniformRBFE(new_x,24, normalize=true) #value 24 and 26 are form lcurve on 400 points
rbf_yn = UniformRBFE(new_y,26, normalize=true)
bfa_xn = BasisFunctionApproximation(xx,new_x,rbf_xn,1)
bfa_yn = BasisFunctionApproximation(yy,new_y,rbf_yn,1)

x_hat=bfa_xn(new_x)
y_hat=bfa_yn(new_y)

using Plots
scatter(new_x,xx, lab="x (ODE slo)")
scatter!(new_x,x_hat, lab="x_hat (RBF apporx.)")
scatter!(new_y,yy,lab="y (ODE slo)")
scatter!(new_y,y_hat, lab="y_hat (RBF apporx.)")    
