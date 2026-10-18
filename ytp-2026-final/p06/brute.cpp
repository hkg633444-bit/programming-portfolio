#include <bits/stdc++.h>
using namespace std;
int n,m,k;
vector<vector<pair<int,int>>> g;
vector<int> cur;
vector<bool> vis;
int best=-1;
void dfs(int u){
    if(u==n){
        vector<int> w=cur;
        sort(w.rbegin(),w.rend());
        int danger=(int)w.size()>k?w[k]:0;
        if(best==-1||danger<best)best=danger;
        return;
    }
    for(auto&e:g[u]){
        if(vis[e.first])continue;
        vis[e.first]=true;
        cur.push_back(e.second);
        dfs(e.first);
        cur.pop_back();
        vis[e.first]=false;
    }
}
int main(){
    cin>>n>>m>>k;
    g.assign(n+1,{});
    for(int i=0;i<m;i++){
        int u,v,w;
        cin>>u>>v>>w;
        g[u].push_back({v,w});
        g[v].push_back({u,w});
    }
    vis.assign(n+1,false);
    vis[1]=true;
    dfs(1);
    cout<<best<<"\n";
}
