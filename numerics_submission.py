"""Reproduce the numerical results in the submission manuscript

Metric-Aware Quantum Linear Solvers for Quasi-Hermitian Systems:
Conditioning, Optimal Metrics, and Exceptional-Point Obstructions
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parent
FIG, TAB, DATA = ROOT/"figures", ROOT/"tables", ROOT/"data"
for d in (FIG, TAB, DATA): d.mkdir(exist_ok=True)
EPS_QSVT = 1e-6
RNG_SEED = 20260924

def cond2(A): return float(np.linalg.cond(A, 2))
def psd_sqrt(A):
    w,U=la.eigh(A)
    if np.min(w)<=0: raise ValueError("Matrix is not positive definite")
    return (U*np.sqrt(w))@U.conj().T
def pt_dimer(g):
    H=np.array([[1j*g,1.],[1.,-1j*g]],complex)
    eta=np.array([[1.,-1j*g],[1j*g,1.]],complex)
    rho=psd_sqrt(eta); h=rho@H@la.inv(rho)
    return H,eta,rho,h
def spectral_projector_norms(H):
    vals,left,right=la.eig(H,left=True,right=True); out=[]
    for j in range(len(vals)):
        v,w=right[:,j],left[:,j]
        P=np.outer(v,np.conj(w))/np.vdot(w,v); out.append(la.norm(P,2))
    return vals,np.asarray(out,float)
def q_at_zero(H):
    vals=la.eigvals(H); return float(np.min(np.abs(vals))*la.norm(la.inv(H),2))
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    plt.savefig(FIG/f"{name}.png",dpi=220,bbox_inches="tight"); plt.close()
def write_table(path,header,rows,caption,label,small=False,scriptsize=False):
    with path.open("w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        if scriptsize:f.write("\\scriptsize\n")
        elif small:f.write("\\small\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write("\\begin{tabular}{l"+"r"*(len(header)-1)+"}\n\\toprule\n")
        f.write(" & ".join(header)+" \\\\\n\\midrule\n")
        for r in rows:f.write(" & ".join(r)+" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

def run_pt():
    gs=np.linspace(0,0.995,300); k=(1+gs)/(1-gs); kr=np.sqrt(k)
    plt.figure(figsize=(6.4,4.2)); plt.semilogy(gs,k,label=r"$\kappa_2(H_\gamma)=\kappa_2(\eta_\gamma)$")
    plt.semilogy(gs,kr,"--",label=r"$\kappa_2(\rho_\gamma)$"); plt.semilogy(gs,np.ones_like(gs),":",label=r"$\kappa_2(h_\gamma)$")
    plt.xlabel(r"$\gamma$"); plt.ylabel("condition number"); plt.legend(); plt.grid(True,which="both",alpha=.25); savefig("pt_conditioning")
    plt.figure(figsize=(6.4,4.2)); plt.semilogy(gs,k,label=r"$\kappa_{\eta,\star}(H_\gamma)$")
    plt.semilogy(gs,k,"--",label=r"$\mathcal{Q}(H_\gamma)^2$"); plt.semilogy(gs,1/(1-gs**2),":",label=r"$\max_j\|P_j\|_2^2$")
    plt.xlabel(r"$\gamma$"); plt.ylabel("lower-bound / optimal-metric scale"); plt.legend(); plt.grid(True,which="both",alpha=.25); savefig("pt_pseudospectral_bounds")
    DH=k*np.log(k/EPS_QSVT); Dh=np.log(1/EPS_QSVT)*np.ones_like(gs)
    plt.figure(figsize=(6.4,4.2)); plt.semilogy(gs,DH,label="direct $H$ / composed $h$"); plt.semilogy(gs,Dh,"--",label="direct norm-normalized $h$")
    plt.xlabel(r"$\gamma$"); plt.ylabel(r"degree proxy $\mathfrak{D}$"); plt.legend(); plt.grid(True,which="both",alpha=.25); savefig("pt_qsvt_degree_proxy")
    csvrows=[]; texrows=[]
    for g in [0.2,0.5,0.8,0.95,0.99]:
        H,eta,rho,h=pt_dimer(g); Q=q_at_zero(H); _,pn=spectral_projector_norms(H)
        row=dict(gamma=g,kappa_H=cond2(H),kappa_h=cond2(h),kappa_eta=cond2(eta),kappa_rho=cond2(rho),Q=Q,Q2=Q**2,Pmax2=float(np.max(pn)**2)); csvrows.append(row)
        texrows.append([f"{g:.2f}",f"{row['kappa_H']:.4g}",f"{row['kappa_h']:.3g}",f"{row['kappa_eta']:.4g}",f"{row['Q2']:.4g}",f"{row['Pmax2']:.4g}"])
    with (DATA/"pt_verification.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=csvrows[0].keys()); w.writeheader(); w.writerows(csvrows)
    write_table(TAB/"pt_verification.tex",[r"$\gamma$",r"$\kappa_2(H)$",r"$\kappa_2(h)$",r"$\kappa_2(\eta)$",r"$\mathcal Q(H)^2$",r"$\max_j\|P_j\|_2^2$"],texrows,"Numerical verification for the PT-symmetric dimer.","tab:pt-verification")

def hatano(N,g,nu=None):
    if nu is None:nu=1/(N+1.)**2
    h=(2+nu)*np.eye(N)-np.eye(N,k=1)-np.eye(N,k=-1)
    H=(2+nu)*np.eye(N)-np.exp(-g)*np.eye(N,k=1)-np.exp(g)*np.eye(N,k=-1)
    rho=np.diag(np.exp(-g*np.arange(N,dtype=float)))
    return H,h,rho,nu
def bound(N,g,nu=None):
    if nu is None:nu=1/(N+1.)**2
    t=np.arccosh(1+nu/2); return float(np.exp(-g*(N-1))*np.sinh((N+1)*t)/np.sinh(t))
def tcount(eps):
    eps=min(max(float(eps),1e-300),.5); return max(0.,3*np.log2(1/eps))

def run_hatano():
    Ns=np.array([8,16,32,64,128]); gvals=[.05,.10,.15]
    plt.figure(figsize=(6.4,4.2))
    for g in gvals:
        s=[]; b=[]
        for N in Ns:
            H,_,_,nu=hatano(int(N),g); s.append(la.svdvals(H)[-1]); b.append(bound(int(N),g,nu))
        plt.semilogy(Ns,s,marker="o",label=rf"$\sigma_{{\min}}(H_{{\rm HN}})$, $g={g:.2f}$"); plt.semilogy(Ns,b,"--",alpha=.8)
    sh=[la.svdvals(hatano(int(N),.1)[1])[-1] for N in Ns]
    plt.semilogy(Ns,sh,marker="s",linestyle=":",linewidth=2,label=r"$\sigma_{\min}(h_N)$"); plt.xlabel(r"system size $N$"); plt.ylabel("smallest singular value"); plt.legend(fontsize=8); plt.grid(True,which="both",alpha=.25); savefig("hatano_nelson_smin_collapse")
    plt.figure(figsize=(6.4,4.2))
    for g in gvals:
        vals=[]
        for N in Ns:
            H,_,_,nu=hatano(int(N),g); vals.append(3*max(2+nu,np.exp(g),np.exp(-g))/la.svdvals(H)[-1])
        plt.semilogy(Ns,vals,marker="o",label=rf"native sparse $H_{{\rm HN}}$, $g={g:.2f}$")
    vh=[]
    for N in Ns:
        _,h,_,nu=hatano(int(N),.1); vh.append(3*(2+nu)/la.svdvals(h)[-1])
    plt.semilogy(Ns,vh,marker="s",linestyle="--",linewidth=2,label=r"direct sparse $h_N$"); plt.xlabel(r"system size $N$"); plt.ylabel(r"effective inverse scale $\overline{\kappa}$"); plt.legend(fontsize=8); plt.grid(True,which="both",alpha=.25); savefig("hatano_nelson_effective_scale")
    rows=[]; trows=[]; ratios=[]; tratios=[]; csvrows=[]
    for N in [8,16,32,64,128,256]:
        H,h,rho,nu=hatano(N,.1); rinv=np.exp(.1*np.arange(N)); bvec=np.zeros(N,complex); bvec[0]=1; rb=rho@bvec; y=la.solve(h,rb); x=rinv*y
        sh=la.svdvals(h)[-1]; sH=la.svdvals(H)[-1]; ah=3*(2+nu); aH=3*max(2+nu,np.exp(.1),np.exp(-.1)); kh=ah/sh; kH=aH/sH
        Dh=kh*np.log(kh/EPS_QSVT); DH=kH*np.log(kH/EPS_QSVT); chi=np.exp(.1*(N-1))*la.norm(y)/la.norm(x); zh=min(1,float(sh*la.norm(y)/la.norm(rb))); zH=min(1,float(sH*la.norm(x)/la.norm(bvec)))
        reps=chi/zh; Qs=Dh*reps; QH=DH/zH; ratio=Qs/QH; ratios.append(ratio); n=int(round(np.log2(N))); Ts=28*n
        Td=QH*(Ts+tcount(EPS_QSVT/max(20*QH,1))); Tsolve=Qs*(Ts+tcount(EPS_QSVT/max(20*Qs,1))); inv=(np.pi/2)*2*reps; rots=max(1,inv*2*n); Tm=rots*tcount(EPS_QSVT/(20*rots)); Tstage=Tsolve+Tm; tr=Tstage/Td; tratios.append(tr)
        rows.append([str(N),f"{sh:.3e}",f"{sH:.3e}",f"{bound(N,.1,nu):.3e}",f"{kh:.4g}",f"{kH:.4g}",f"{chi:.4g}",f"{zh:.4g}",f"{ratio:.3e}"]); trows.append([str(N),f"{Qs:.3e}",f"{QH:.3e}",f"{Tm:.3e}",f"{Tstage:.3e}",f"{Td:.3e}",f"{tr:.3e}"])
        csvrows.append(dict(N=N,smin_h=sh,smin_H=sH,smin_H_upper_bound=bound(N,.1,nu),kbar_h=kh,kbar_H=kH,chi_out=chi,zeta_h=zh,Q_stage=Qs,Q_H=QH,Q_stage_over_Q_H=ratio,T_metric=Tm,T_stage=Tstage,T_direct_H=Td,T_stage_over_T_H=tr))
    plt.figure(figsize=(6.4,4.2)); Ns2=np.array([8,16,32,64,128,256]); plt.semilogy(Ns2,ratios,marker="o",label="coherent plain-QSVT proxy"); plt.semilogy(Ns2,tratios,marker="s",linestyle="--",label="leading logical $T$-count ratio"); plt.axhline(1,linewidth=1,linestyle=":"); plt.xlabel(r"system size $N$"); plt.ylabel("staged / native-direct resource ratio"); plt.legend(); plt.grid(True,which="both",alpha=.25); savefig("hatano_nelson_resource_ratio")
    write_table(TAB/"hatano_nelson_resources.tex",[r"$N$",r"$\sigma_{\min}(h)$",r"$\sigma_{\min}(H)$","analytic bound",r"$\overline{\kappa}_h$",r"$\overline{\kappa}_H$",r"$\chi_{\rm out}$",r"$\zeta_h$","staged/direct ratio"],rows,"Shifted open-boundary Hatano--Nelson chain at $g=0.1$ and $\epsilon=10^{-6}$.","tab:hatano-nelson-resources",scriptsize=True)
    write_table(TAB/"hatano_nelson_tcounts.tex",[r"$N$",r"$\widehat Q_{\rm stage}$",r"$\widehat Q_H$","metric $T$","staged $T$","direct-$H$ $T$",r"$T_{\rm stage}/T_H$"],trows,"Leading logical Clifford+$T$ accounting for the shifted Hatano--Nelson benchmark at $g=0.1$.","tab:hatano-nelson-tcounts",small=True)
    with (DATA/"hatano_nelson_resources.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=csvrows[0].keys()); w.writeheader(); w.writerows(csvrows)

if __name__=="__main__":
    print(f"Reproducibility seed: {RNG_SEED}"); print(f"QSVT proxy tolerance: {EPS_QSVT:g}")
    run_pt(); run_hatano(); print("Numerical reproduction complete.")
