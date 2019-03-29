using DifferentialEquations
using Optim
function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - u[1]*u[2]
  du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)

sol = solve(prob,Tsit5())
t = collect(range(0,stop=10,length=400))


using RecursiveArrayTools
randomized = VectorOfArray([(sol(t[i])) for i in 1:length(t)])
data = convert(Array,randomized)

x=Vector{Float64}(undef,400)
y=Vector{Float64}(undef,400)

for i in range(1,stop=400)
  x[i]=data[1,i]
end

for i in range(1,stop=400)
  y[i]=data[2,i]
end


using BasisFunctionExpansions

nv=2:100

lcurve_x = map(nv) do n
  rbf = UniformRBFE(t, n, normalize = true)
  bfa = BasisFunctionApproximation(x,t,rbf,1)
  (x-bfa(t))
end

lcurve_y = map(nv) do n
  rbf = UniformRBFE(t, n, normalize = true)
  bfa = BasisFunctionApproximation(y,t,rbf,1)
  (y-bfa(t))
end

i=1
j=1
x_err = Array{Float64}(undef, 99)
for i in range(1,stop=99)
  d=0
  for j in range(1,stop=400)
    d=(lcurve_x[i][j])^2+d
  x_err[i]=(d^0.5)
  end
end
y_err = Array{Float64}(undef, 99)
for i in range(1,stop=99)
  d=0
  for j in range(1,stop=400)
    d=(lcurve_y[i][j])^2+d
  y_err[i]=(d^0.5)
  end
end

function f(d)
  i=1
  j=2
  h=999
  for i in range(1,stop=99)
    if d[i]<=h
      h=d[i]
      j=i
    end
  end
  return j
end

min_x=f(x_err)
min_y=f(y_err)

rbf_x = UniformRBFE(t,min_x, normalize=true)
rbf_y = UniformRBFE(t,min_y, normalize=true)
bfa_x = BasisFunctionApproximation(x,t,rbf_x,1)
bfa_y = BasisFunctionApproximation(y,t,rbf_y,1)

x_hat_new  = bfa_x(t)
y_hat_new  = bfa_y(t)




##################### D value##################
# finding Dmin ,Dmax for individual candidate point

Dis=Vector{Float64}(undef,400)

function d_c(rp,t)
  for i in range(1,stop=length(t))
    if (t[i]-rp)>0
      Dis[i]=t[i]-rp
    else
      Dis[i]=rp-t[i]
    end
  end
  dn=11
  for i in range(1,stop=length(t))
    if Dis[i]<dn
      dn=Dis[i]
    end
  end
  for i in range(1,stop=length(t))
    dm=-1
    if Dis[i]>dm
      dm=Dis[i]
    end
  end
  return dm,dn
end

#incumbent point for x
function incp(x)
  icp=10
  for i in range(1,stop=length(t))
    if x[i]<icp
      icp=x[i]
    end
  end
  return icp
end


#B is the candidate point array & finding Global Dmin,Dmax

function DMin_Max(B,t)
  for i in range(1,stop=50)  #size of array instead of 50 # genarating candidate point each from B
    a=B[i]
    g=11
    h=-1
    m,n= d_c(a,t)
    if m>h
      h=m
    end
    if n <g
      g=n
    end
  end
  return m,n
end

function d_n(rp_at_obj,t) # candidate point at object evalution
  h=11
  for i in range(1,stop=length(t))
    if (t[i]-rp_at_obj)>0
      if (t[i]-rp_at_obj)<h
        h=t[i]-rp_at_obj
      end
    else
       (rp_at_obj-t[i])<h
       h=rp_at_obj-t[i]
    end
  end
  return h
end



function D(x,t,B)
  m,n=DMin_Max(B,t)
  p=(m-d_n(x,t))/(m-n)
  return p
end

######## --------------#########



function fxn(t)
  h=10
  for i in range(1,stop=length(t))
    if bfa_x(t[i]) < h
      h=bfa_x(t[i])
    end
  end
  return h
end

function fxm(t)
  l=0
  for i in range(1,stop=length(t))
    if bfa_x(t[i])>h
      l=bfa_x(t[i])
    end
  end
  return l
end


function S(f,t)
  s=(bfa_x(f)-fxn(t))/(fxm(t)-fnx(t))
  return s
end

#########______________############


# randomly generate t points near incumbent point “a"


function random_x(a)
  CP=Vector{Float64}(undef,50)
                               #alpha=.1
  for i in range(1,stop=50)

   CP[i]= a+ 0.1rand()
  end
  return CP
end

########-----------##############

function Obj(x,t)  # x[] is the original function values, evaluated by costly function ODE solver
  a=incp(x)
  as=random_x(a)  # as is the array of candidate points
  w= .2
  c=0         # select the value of  w .2  .4  .8
  for i in range(1,stop=50)  #stop value
    u = w*S(as[i],t) + (1-w)*D(as[i],t,as)  #as[i] candidate( time) array on x-axis t is time for evaluated points
    println("$u")
    #G=maximum
    #if u<G
    #  G=u
    #  c=x
    #end
  end
  #return c
  println("$c")
end

Obj(x,t)




scatter(t,x, lab="x (ODE slo)")
scatter!(t,x_hat_new, lab="x_hat (RBF apporx.)")
scatter!(t,y,lab="y (ODE slo)")
scatter!(t,y_hat_new, lab="y_hat (RBF apporx.)")
