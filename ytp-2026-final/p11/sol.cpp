#include <bits/stdc++.h>
using namespace std;
#define ll long long
typedef unsigned long long ull;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    ll k;
    cin>>n>>k;
    vector<array<ll,5>> p(n);
    for(auto&q:p)for(int d=0;d<5;d++)cin>>q[d];
    int nw=(n+63)/64;
    const int step=64;
    vector<vector<int>> ord(5,vector<int>(n));
    vector<vector<int>> lo(5,vector<int>(n)),hi(5,vector<int>(n));
    vector<vector<ull>> ck(5);
    int nc=n/step+1;
    for(int d=0;d<5;d++){
        iota(ord[d].begin(),ord[d].end(),0);
        sort(ord[d].begin(),ord[d].end(),[&](int a,int b){return p[a][d]<p[b][d];});
        vector<ll> vals(n);
        for(int i=0;i<n;i++)vals[i]=p[ord[d][i]][d];
        for(int i=0;i<n;i++){
            int id=ord[d][i];
            ll v=p[id][d];
            lo[d][id]=lower_bound(vals.begin(),vals.end(),v-k)-vals.begin();
            hi[d][id]=upper_bound(vals.begin(),vals.end(),v+k)-vals.begin();
        }
        ck[d].assign((size_t)nc*nw,0);
        vector<ull> curb(nw,0);
        for(int c=0;c<nc;c++){
            memcpy(&ck[d][(size_t)c*nw],curb.data(),nw*sizeof(ull));
            for(int i=c*step;i<min(n,(c+1)*step);i++){
                int id=ord[d][i];
                curb[id>>6]|=1ULL<<(id&63);
            }
        }
    }
    ll total=0;
    vector<ull> res(nw),tmp(nw);
    for(int i=0;i<n;i++){
        for(int d=0;d<5;d++){
            int l=lo[d][i],r=hi[d][i];
            vector<ull>&dst=(d==0)?res:tmp;
            const ull*pa=&ck[d][(size_t)(r/step)*nw];
            const ull*pb=&ck[d][(size_t)(l/step)*nw];
            for(int w=0;w<nw;w++)dst[w]=pa[w]^pb[w];
            for(int t=(r/step)*step;t<r;t++){
                int id=ord[d][t];
                dst[id>>6]^=1ULL<<(id&63);
            }
            for(int t=(l/step)*step;t<l;t++){
                int id=ord[d][t];
                dst[id>>6]^=1ULL<<(id&63);
            }
            if(d>0)for(int w=0;w<nw;w++)res[w]&=tmp[w];
        }
        for(int w=0;w<nw;w++)total+=__builtin_popcountll(res[w]);
    }
    cout<<(total-n)/2<<"\n";
    return 0;
}
