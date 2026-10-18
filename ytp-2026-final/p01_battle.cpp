#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pb push_back
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll h1,h2;
    int n,m;
    cin>>h1>>h2;
    vector<ll> d1(101,0),d2(101,0);
    cin>>n;
    for(int i=0;i<n;i++){
        ll d;int c;
        cin>>d>>c;
        d1[c]+=d;
    }
    cin>>m;
    for(int i=0;i<m;i++){
        ll d;int c;
        cin>>d>>c;
        d2[c]+=d;
    }
    vector<int> c1,c2;
    for(int c=1;c<=100;c++){
        if(d1[c])c1.pb(c);
        if(d2[c])c2.pb(c);
    }
    const int lim=10000200;
    vector<int> a1(lim+1,0),a2(lim+1,0);
    for(int c:c1)for(int t=c;t<=lim;t+=c)a1[t]+=(int)d1[c];
    for(int c:c2)for(int t=c;t<=lim;t+=c)a2[t]+=(int)d2[c];
    for(int t=1;t<=lim;t++){
        h2-=a1[t];
        if(h2<=0){cout<<"1\n";return 0;}
        h1-=a2[t];
        if(h1<=0){cout<<"2\n";return 0;}
    }
    return 0;
}
