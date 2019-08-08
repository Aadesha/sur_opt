
#=
source
http://www.ressources-actuarielles.net/EXT/ISFA/1226.nsf/9c8e3fd4d8874d60c1257052003eced6/e7dc33e4da12c5a9c12576d8002e442b/$FILE/Jones01.pdf
=#
using DifferentialEquations
using Plots
using BasisFunctionExpansions

#ODE Equation

function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - u[1]*u[2]
  du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]

#solving ODE

prob = ODEProblem(f,u0,tspan,p)

sol = solve(prob,Tsit5())

vv = collect(range(0,stop=10,length=50))

#creating sample data

using RecursiveArrayTools
randomized = VectorOfArray([(sol(vv[i])) for i in 1:length(vv)])
data = convert(Array,randomized)

x=[data[1,i] for i in range(1,stop=length(vv))]

y=[data[2,i] for i in range(1,stop=length(vv))]


#_________________#
#fitting the surface over data

function Rbff(cp)
    rbf_x = UniformRBFE(vv,24, normalize=true)
    rbf_y = UniformRBFE(vv,26, normalize=true)
    bfa_x = BasisFunctionApproximation(x,vv,rbf_x,1)
    bfa_y = BasisFunctionApproximation(y,vv,rbf_y,1)

    x_hat  = bfa_x(cp)
    y_hat  = bfa_y(cp)
    aa=minimum(x_hat)
    bb=minimum(y_hat)
    ee=maximum(x_hat)
    ff=maximum(y_hat)
    return aa,bb,x_hat,y_hat,ee,ff
end

#___D_function_____________#

function d_c(cp,q)
Dis=Vector{Float64}(undef,50)
  for k in range(1,stop=length(cp))
    if (cp[k]-cp[q])>0
      Dis[k]=cp[k]-cp[q]
    else
      Dis[k]=cp[q]-cp[k]
    end
  end
  return maximum(Dis),minimum(Dis)
end

function DMin_Max(cp)
  g=11
  h=-1

  for i in range(1,stop=50)
    a=cp[i]
    m,n= d_c(cp,i)
    if m>h
      h=m
    end
    if n <g
      g=n
    end
  end
  return h,g
end

function d_n(cp,e)
  h=11

  for i in range(1,stop=length(cp))
    if (cp[i]-cp[e])>0
      if (cp[i]-cp[e])<h
        h=cp[i]-cp[e]
      end
    else
      if cp[e]-cp[i]<h
         h=cp[e]-cp[i]
      end
    end
  end
  return h
end

function D(cp,i)
  m,n=DMin_Max(cp)
  p=(m-d_n(cp,i))/(m-n)
  return p
end

#_____S_function_____________#

function S(CP,i,lb)

  aa,bb,x_hat,y_hat,ee,ff=Rbff(CP)
  if lb==0
    s=(x_hat[i]-aa)/(ee-aa)
    return s
  end
  if lb==1
    s=(y_hat[i]-bb)/(ff-bb)
    return s
  end
end

#__incumbent point__________#


function incp(lb)
  if lb==0
    icp=10
    a1=1
   for i in range(1,stop=length(x))
      if x[i]<=icp
        icp=x[i]
        a1=i
      end
    end
    println(icp)
    return icp,vv[a1]
  end
  if lb==1
    icp=10
    a1=1
    for i in range(1,stop=length(y))
      if y[i]<=icp
        icp=y[i]
        a1=i
      end
    end
    return icp,vv[a1]
  end
end

#_____ randomly generate t points near incumbent point “a" --#
function randomso(a)
  CP =[ (a + .01rand()) for i in range(1,stop=25)]#alpha=.1
  GP =[ (a - .01rand()) for i in range(1,stop=25)]
  append!(CP,GP)
  sort!(CP)
  return CP
end

#_Objective function to optimise______#

function Obj(w,lb)
  s,t1=incp(lb)
  CP=randomso(t1)#CP is the array of candidate points
  G=0
  # select the value of  w =.2/.4/.95
  u=[ (w*S(CP,i,lb) + (1-w)*D(CP,i)) for i in range(1,stop=50)]
  b=minimum(u)
  for i in range(1,stop=length(u))
    if u[i]==b
      G=i #to mark index of time at which u is min
    end
  end
  return CP[G] #time
end

#__________#

function A(n,lb) # n number of sampling #lb=0 then optimise on x lb=1 optimse on y for minimum
  for i in range(1,stop=n)
    w=.2
    if i==n/3
      w=.4
    end
    if i==(2*n)/3
      w=.95
    end
    k=Obj(w,lb)
    d,d1=sol(k)
     #update the x,y,vv(time)
    append!(x,d)
    append!(y,d1)
    append!(vv,k)
    u,e=incp(lb)
    println(u) #print the optimal minimum value lb=0 for x,lb=1 for y for each iteration
  end
end
# for example
println("optimise for Y")
A(30,1) # for example
println("optimise for X")
A(30,0) 
