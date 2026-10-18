#include <bits/stdc++.h>
using namespace std;
int main(){
    int n;
    long long k;
    cin>>n>>k;
    vector<array<long long,5>> p(n);
    for(auto&q:p)for(int d=0;d<5;d++)cin>>q[d];
    long long ans=0;
    for(int i=0;i<n;i++)for(int j=i+1;j<n;j++){
        long long mx=0;
        for(int d=0;d<5;d++)mx=max(mx,llabs(p[i][d]-p[j][d]));
        if(mx<=k)ans++;
    }
    cout<<ans<<"\n";
}
