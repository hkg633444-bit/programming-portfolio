#include <bits/stdc++.h>
using namespace std;
#define pb push_back
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n,m,k;
    cin>>n>>m>>k;
    vector<int> eu(m),ev(m),ew(m);
    vector<vector<pair<int,int>>> g(n+1);
    vector<int> ws;
    ws.pb(0);
    for(int i=0;i<m;i++){
        cin>>eu[i]>>ev[i]>>ew[i];
        g[eu[i]].pb({ev[i],ew[i]});
        g[ev[i]].pb({eu[i],ew[i]});
        ws.pb(ew[i]);
    }
    sort(ws.begin(),ws.end());
    ws.erase(unique(ws.begin(),ws.end()),ws.end());
    const int inf=1e9;
    vector<int> dist(n+1);
    auto ok=[&](int lim)->bool{
        fill(dist.begin(),dist.end(),inf);
        deque<int> dq;
        dist[1]=0;
        dq.pb(1);
        while(!dq.empty()){
            int u=dq.front();dq.pop_front();
            for(auto&e:g[u]){
                int c=e.second>lim?1:0;
                if(dist[u]+c<dist[e.first]){
                    dist[e.first]=dist[u]+c;
                    if(c)dq.pb(e.first);
                    else dq.push_front(e.first);
                }
            }
        }
        return dist[n]<=k;
    };
    if(!ok(ws.back())){
        cout<<"-1\n";
        return 0;
    }
    int lo=0,hi=(int)ws.size()-1;
    while(lo<hi){
        int mid=(lo+hi)/2;
        if(ok(ws[mid]))hi=mid;
        else lo=mid+1;
    }
    cout<<ws[lo]<<"\n";
    return 0;
}
