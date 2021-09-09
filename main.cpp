#include "bits/stdc++.h"
using namespace std;
using ll = long long;
using pii = pair<int,int>;
using pll = pair<ll,ll>;
template<typename T>
int sz(const T &a){return int(a.size());}
const string testdata="mnist_check.txt";
const string storage="storage.txt";
const string result_file="out.txt";
const int seed=8239;
const int layers=2;
const int nodes=8;
const int inputsize=28*28;
const int outputsize=10;
const int datasize=10000;
const int samplesize=5;
const double stop=0.1;
const int sizeofcheck=1000;
const double maxvalue=254;
const double betarate=0.9;
const double alpharate=0.1;
const double MV=1;

mt19937 gen(seed);
uniform_real_distribution<double> dist(-MV,MV);
uniform_int_distribution<int> sampleselect(0,datasize-1);
double sigmoid(double a){
    return 1/(1+exp(-a));
}
double dsigmoid(double a){
    return sigmoid(a)*(1-sigmoid(a));
}
void softmax(vector<double>& a){
    double sum=0;
    for(auto x:a)sum+=exp(x);
    for(auto &&x:a)x=exp(x)/sum;
}
double cost(const vector<double> &out, const vector<double> &exp){
    double tot=0;
    for(int i=0;i<sz(out);i++){
        tot+=(exp[i]-out[i])*(exp[i]-out[i]);
    }
    return tot;
}

/*
main reference used: 3blue1brown video and https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9
mnist data procceser from https://jamesmccaffrey.wordpress.com/2017/10/16/getting-mnist-data-into-a-text-file/ 
gradientchecking from https://towardsdatascience.com/how-to-debug-a-neural-network-with-gradient-checking-41deec0357a9

28*28 input nodes, 10 output nodes
2 hidden layers, 32 nodes each
sigmoid function for all nodes
the cost of one example is sum of squared differences (exp-given)^2
cost of one set of examples is average cost of examples

Each example gives a nudge amount to every bias and weight, the amount that is applied
is equal to the average of all the nudges over all examples in the set.

derivative for each weight/bias wrt to the single training example,
derivative of full function is then jkust average derivative of all training functions (since cost is averaged)

gradient is the vector of the derivatives (how every input affects the cost function)
each deriative tells you if we increase this lets say by 1, how much does that change cost function,
    so if we want to go up as fast as possible, we would increase by that much. (ie imagine the slope is +, increasing value increases cost, so you increase value, and vice versa)
    thus, to lower as much as possible, we would go in the opposite direction, - of that.

costfunc(given)=(exp-given)^2+a

cost(bias) = costfunc(toinner(bias))

d/dx cost(bias)/bias

= d/dx costfunc(toinner(bias))/toinner(bias) * d/dx toinner(bias)/bias

d/dx costfunc(given)=-2(exp-given)=2(given-exp)

= 2(given-exp) * 
{
    toinner(bias) = sigmoid(addition(bias))
    d/dx toinner(bias)/bias 
    = sigmoid'(addition(bias)) * d/dx addition(bias)/bias
    = sigmoid'(addition(bias))



}


d/dx of valuebeforesigmoid wrt output value of node = sigmoid'(valuebeforesigmoid)
cost(valuebeforesigmoid)=cost(sigmoid(valuebeforesigmoid))
d/dx of valuebeforesigmoid wrt cost = d/dx cost(outputofnode)/outputofnode * sigmoid'(valuebeforesigmoid)

d/dx bias wrt cost = d/dx valuebeforesigmoid wrt cost

cost(x) = costfrombeforesigmoid(x*value+otherstuff)

d/dx cost(x) = costfrombeforesigmoid'(x*value+otherstuff) * value = d/dx of valuebeforesigmoid wrt cost * value

d/dx of a value wrt cost =  d/dx of valuebeforesigmoid wrt cost * weight




need value before sigmoid





Todo Later?
    softmax function?


*/

struct network{
    struct node{
        vector<double> weights;
        double bias;
        double bsig,asig;
        node(){
            weights=vector<double>();
            bias=0;
            bsig=0,asig=0;
        }
        double calc(const vector<double> &values){
            double value=bias;
            for(int i=0;i<sz(values);i++)value+=values[i]*weights[i];
            bsig=value;
            return asig=sigmoid(value); 
        }
    };
    int paramcnt;
    vector<node> arr[layers+2];
    vector<double> velocity;
    network(int start, int end){
        paramcnt=nodes*layers+end+nodes*start+nodes*end+nodes*nodes*(layers-1);
        arr[0].resize(start);
        arr[layers+1].resize(end);
        for(int i=1;i<=layers+1;i++){
            if(i<=layers)arr[i].resize(nodes);
            for(int j=0;j<sz(arr[i]);j++){
                arr[i][j].bias=dist(gen);
                arr[i][j].weights.resize((i==0?start:sz(arr[i-1])));
                for(int k=0;k<sz(arr[i][j].weights);k++){
                    arr[i][j].weights[k]=dist(gen);
                }
            }
        }
    }
    void get(){
        ifstream store(storage);
        for(int i=1;i<=layers+1;i++){
            for(auto &&x:arr[i]){
                store>>x.bias;
                for(int j=0;j<sz(x.weights);j++)store>>x.weights[j];
            }
        }
        store.close();
    }
    void print(){
        ofstream store(storage);
        for(int i=1;i<=layers+1;i++){
            for(auto x:arr[i]){
                store<<setprecision(9)<<x.bias<<" ";
                for(int j=0;j<sz(x.weights);j++)store<<setprecision(9)<<x.weights[j]<<" \n"[j==sz(x.weights)-1];
            }
        }
        store.close();
    }
    pair<int,vector<double>> solve(const vector<double> &input){
        vector<double> prev=input;
        for(int i=0;i<sz(input);i++)arr[0][i].asig=input[i];
        for(int i=1;i<=layers+1;i++){
            vector<double> te;
            for(auto &&cur:arr[i])te.push_back(cur.calc(prev));
            prev=te;
        }
        return {max_element(prev.begin(),prev.end())-prev.begin(),prev};
    }
    vector<double> backprop(vector<double> deriv){//deriv of output of every node wrt cost
        vector<double> gradient;
        for(int i=layers+1;i>=1;i--){
            vector<double> nderiv(sz(arr[i-1]),0);
            for(int j=0;j<sz(arr[i]);j++){
                double derivbeforesig=dsigmoid(arr[i][j].bsig)*deriv[j];
                gradient.push_back(derivbeforesig);
                for(int k=0;k<sz(arr[i][j].weights);k++){
                    gradient.push_back(derivbeforesig*arr[i-1][k].asig);
                    nderiv[k]+=derivbeforesig*arr[i][j].weights[k];
                }
            }
            deriv=nderiv;
        }
        return gradient;
    }
    double applygradient(vector<double> gradient){
        int ptr=0;
        double largestchange=0;
        for(int i=layers+1;i>=1;i--){
            for(int j=0;j<sz(arr[i]);j++){
                velocity[ptr]=velocity[ptr]*betarate+(1-betarate)*gradient[ptr];
                largestchange=max(largestchange,abs(alpharate*velocity[ptr]));
                arr[i][j].bias-=alpharate*velocity[ptr++];
                for(int k=0;k<sz(arr[i][j].weights);k++){
                    velocity[ptr]=velocity[ptr]*betarate+(1-betarate)*gradient[ptr];
                    largestchange=max(largestchange,abs(alpharate*velocity[ptr]));
                    arr[i][j].weights[k]-=alpharate*velocity[ptr++];
                }
            }
        }
        return largestchange;
    }
    void train(const vector<vector<double>> &input, const vector<vector<double>> &exp){
        velocity=vector<double>(paramcnt,0);
        int am=0;
        int lastworse=0;
        while(1){
            vector<double> totgradient(paramcnt,0);
            double totcost=0;
            for(int i=0;i<samplesize;i++){
                int cur=sampleselect(gen);
                auto result=solve(input[cur]).second;
                totcost+=cost(result,exp[cur]);
                for(int j=0;j<sz(result);j++)result[j]=2*(result[j]-exp[cur][j]);
                vector<double> gradient=backprop(result);
                transform(totgradient.begin(),totgradient.end(),gradient.begin(),totgradient.begin(),plus<>{});
            }
            for(auto &&x:totgradient)x/=(double)samplesize;
            am++;
            double change=applygradient(totgradient);
            totcost/=samplesize;
            if(am%100==0){
                print();
                printf("%.9f %.9f\n",totcost,change);
            }
            if(totcost>stop)lastworse=am;
            if(am-lastworse>sizeofcheck)break;
        }
    }
    double gradientchecking(const vector<vector<double>> &input, const vector<vector<double>> &exp, double epsilon){
        double worsterror=0,expected,given;
        int data,layer,node,value;
        for(int cur=0;cur<sz(input);cur++){
            auto result=solve(input[cur]).second;
            for(int j=0;j<sz(result);j++)result[j]=2*(result[j]-exp[cur][j]);
            vector<double> gradient=backprop(result);
            int ptr=0;
            for(int i=layers+1;i>=1;i--){
                for(int j=0;j<sz(arr[i]);j++){
                    double approxderiv=0;
                    arr[i][j].bias+=epsilon;
                    approxderiv+=cost(solve(input[cur]).second,exp[cur]);
                    arr[i][j].bias-=2*epsilon;
                    approxderiv-=cost(solve(input[cur]).second,exp[cur]);
                    arr[i][j].bias+=epsilon;
                    approxderiv/=(2*epsilon);
                    if(abs(approxderiv-gradient[ptr])>worsterror){
                        worsterror=abs(approxderiv-gradient[ptr]);
                        data=cur,layer=i,node=j,value=0;
                        expected=approxderiv,given=gradient[ptr];
                    }
                    ptr++;
                    for(int k=0;k<sz(arr[i][j].weights);k++){
                        approxderiv=0;
                        arr[i][j].weights[k]+=epsilon;
                        approxderiv+=cost(solve(input[cur]).second,exp[cur]);
                        arr[i][j].weights[k]-=2*epsilon;
                        approxderiv-=cost(solve(input[cur]).second,exp[cur]);
                        arr[i][j].weights[k]+=epsilon;
                        approxderiv/=(2*epsilon);
                        if(abs(approxderiv-gradient[ptr])>worsterror){
                            worsterror=abs(approxderiv-gradient[ptr]);
                            data=cur,layer=i,node=j,value=k+1;
                            expected=approxderiv,given=gradient[ptr];
                        }
                        ptr++;
                    }
                }
            }
        }
        printf("OFF: %.9f\nGiven: %.9f\nExpected: %.9f\nTestCase: %d\nLayer: %d\nNode: %d\nValue: %d\n",worsterror,given,expected,data,layer,node,value);
        return worsterror;
    }
};
pair<vector<vector<double>>,vector<vector<double>>> read(string path){
    ifstream data(path);
    string trash;
    double a;
    vector<vector<double>> input,output;
    for(int i=0;i<datasize;i++){
        data>>trash>>a;
        vector<double> exp(10,0);
        exp[a]=1;
        data>>trash;
        vector<double> given;
        for(int j=0;j<inputsize;j++){
            data>>a;
            given.push_back(a/maxvalue);
        }
        input.push_back(given),output.push_back(exp);
    }
    data.close();
    return {input,output};
}

int main(){
    cin.tie(NULL);
    ios_base::sync_with_stdio(false);
    int which;
    cin>>which;
    if(which==0){//testing
        network solver(inputsize,outputsize);
        solver.get();
        string trash;
        double a;
        vector<vector<double>> input,output;
        tie(input,output)=read(testdata);
        int amountright=0;
        vector<pii> toprint(datasize);
        for(int i=0;i<datasize;i++){
            int correct=0;
            for(int j=0;j<outputsize;j++)if(output[i][j]==1)correct=toprint[i].first=j;
            int given=solver.solve(input[i]).first;
            if(given==correct)amountright++;
            toprint[i].second=given;
        }
        ofstream result(result_file);
        result<<amountright<<"/"<<datasize<<"\n";
        for(auto x:toprint)result<<"Exp: "<<x.first<<" Given: "<<x.second<<"\n";
        result.close();
    }
    else if(which==1||which==2){//training
        network solver(inputsize,outputsize);
        if(which==1)solver.get();//1 is use old data, 2 is use new random data.
        vector<vector<double>> input,output;
        tie(input,output)=read(testdata);
        solver.train(input,output);
        solver.print();
    }
    else{//gradient checking
        network solver(inputsize,outputsize);
        // solver.get();
        vector<vector<double>> input,output;
        tie(input,output)=read(testdata);
        solver.gradientchecking(input,output,0.0001);
    }
    return 0;
}