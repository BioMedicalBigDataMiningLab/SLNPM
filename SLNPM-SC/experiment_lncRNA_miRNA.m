function experiment_lncRNA_miRNA
seed=1;
cv_num=5;
neighbor_alpha=0.1:0.1:0.9;
LP_alpha=0.1:0.1:0.9;
lncRNA_alpha=0:0.05:1.0;
for i =1:length(neighbor_alpha)
    for j =1:length(LP_alpha)
        for k =1:length(lncRNA_alpha)
            final_result = CrossValidation(seed, cv_num, neighbor_alpha(i), LP_alpha(j), lncRNA_alpha(k))
            save(strcat('LNSLP_', num2str(neighbor_alpha(i)), '_', num2str(LP_alpha(j)), '_', num2str(lncRNA_alpha(k)), '.mat'), 'final_result');
        end
    end
end
end

function final_result=CrossValidation(seed, cv_num, neighbor_alpha, LP_alpha, lncRNA_alpha)
load('lncRNA_miRNA_all.mat');
interaction_matrix=interactionmatrix;         %lncRNA端
[row_index,col_index]=find(interaction_matrix==1);
link_num=sum(sum(interaction_matrix));
rand('state',seed);
random_index=randperm(link_num);
size_of_cv=round(link_num/cv_num);
result=zeros(1,7);
for k=1:cv_num
    if (k~=cv_num)
        test_row_index=row_index(random_index((size_of_cv*(k-1)+1):(size_of_cv*k)));
        test_col_index=col_index(random_index((size_of_cv*(k-1)+1):(size_of_cv*k)));
    else
        test_row_index=row_index(random_index((size_of_cv*(k-1)+1):end));
        test_col_index=col_index(random_index((size_of_cv*(k-1)+1):end));
    end
    train_interaction_matrix=interaction_matrix;
    test_link_num=size(test_row_index,1);
    for i=1:test_link_num
        train_interaction_matrix(test_row_index(i),test_col_index(i))=0;
    end

    similairty_matrix_1=GetImprovedLNSimilarity(train_interaction_matrix, L_Kmer_5, neighbor_alpha);
    score_matrix_1=LabelPropagation(similairty_matrix_1,train_interaction_matrix,LP_alpha);
    similairty_matrix_2=GetImprovedLNSimilarity(train_interaction_matrix', M_Kmer_5, neighbor_alpha);
    score_matrix_2=LabelPropagation(similairty_matrix_2,train_interaction_matrix',LP_alpha);
    score_matrix = lncRNA_alpha * score_matrix_1 + (1 - lncRNA_alpha) * score_matrix_2';
    result(1,:)=result(1,:)+ModelEvaluate(interaction_matrix,score_matrix,train_interaction_matrix);
end
final_result=result/cv_num;
end

function similarity_matrix=GetImprovedLNSimilarity(train_interaction_matrix,feature_matrix,neighbor_alpha)
index_0_row=find(all(train_interaction_matrix==0,2)==1);

%初始化相似度矩阵
similarity_matrix=ones(size(train_interaction_matrix, 1));
for i=1:size(index_0_row, 1)
    similarity_matrix(index_0_row(i, 1), :)=zeros(1, size(train_interaction_matrix, 1));
    similarity_matrix(:, index_0_row(i, 1))=zeros(size(train_interaction_matrix, 1), 1);
end

%填充反应谱相似度
temp_interaction=train_interaction_matrix;
temp_interaction(index_0_row, :)=[];
temp_similarity_matrix=GetLNSimilarity(temp_interaction, round(size(temp_interaction, 1)*neighbor_alpha));
count = 1;
for j=1:size(similarity_matrix(:), 1)
    if(similarity_matrix(j)==1)
        similarity_matrix(j)=temp_similarity_matrix(count);
        count=count+1;
    end
end

%填充特征相似度
feature_similarity_matrix=GetLNSimilarity(feature_matrix, round(size(feature_matrix, 1)*neighbor_alpha));
for k=1:size(index_0_row, 1)
    similarity_matrix(index_0_row(k, 1), :)=feature_similarity_matrix(index_0_row(k, 1), :);
end
end

function score_matrix=LabelPropagation(W,Y,alpha)
%W==similarity_matrix; Y==train_interaction_matrix
score_matrix=(1-alpha)*pinv(eye(size(W,1))-alpha*W)*Y;
end

function result=ModelEvaluate(interaction_matrix,score_matrix,train_interaction_matrix)
real_label=interaction_matrix(:);
predict_score=score_matrix(:);
index=train_interaction_matrix(:);
test_index=find(index==0);
real_label=real_label(test_index);
predict_score=predict_score(test_index);
aupr=AUPR(real_label,predict_score);
auc=AUC(real_label,predict_score);
[sen,spec,precision,accuracy,f1]=EvaluationMetric(real_label,predict_score);
result=[aupr,auc,sen,spec,precision,accuracy,f1];
end

function [sen,spec,precision,accuracy,f1]=EvaluationMetric(real_label,predict_score)
sorted_predict_score=unique(sort(predict_score));
score_num=size(sorted_predict_score,1);
threshold=sorted_predict_score(ceil(score_num*(1:999)/1000));
for i=1:999
    predict_label=(predict_score>threshold(i));
    [temp_sen(i),temp_spec(i),temp_precision(i),temp_accuracy(i),temp_f1(i)]=ClassificationMetric(real_label,predict_label);
end
[max_score,index]=max(temp_f1);
sen=temp_sen(index);
spec=temp_spec(index);
precision=temp_precision(index);
accuracy=temp_accuracy(index);
f1=temp_f1(index);
end

function [sen,spec,precision,accuracy,f1]=ClassificationMetric(real_label,predict_label)
tp_index=find(real_label==1 & predict_label==1);
tp=size(tp_index,1);

tn_index=find(real_label==0 & predict_label==0);
tn=size(tn_index,1);

fp_index=find(real_label==0 & predict_label==1);
fp=size(fp_index,1);

fn_index=find(real_label==1 & predict_label==0);
fn=size(fn_index,1);

accuracy=(tn+tp)/(tn+tp+fn+fp);
sen=tp/(tp+fn);
recall=sen;
spec=tn/(tn+fp);
precision=tp/(tp+fp);
f1=2*recall*precision/(recall+precision);
end

function area=AUPR(real_label,predict_score)
sorted_predict_score=unique(sort(predict_score));
score_num=size(sorted_predict_score,1);
threshold=sorted_predict_score(ceil(score_num*(1:999)/1000));

threshold=threshold';
threshold_num=length(threshold);
tn=zeros(threshold_num,1);
tp=zeros(threshold_num,1);
fn=zeros(threshold_num,1);
fp=zeros(threshold_num,1);

for i=1:threshold_num
    tp_index=logical(predict_score>=threshold(i) & real_label==1);
    tp(i,1)=sum(tp_index);
    
    tn_index=logical(predict_score<threshold(i) & real_label==0);
    tn(i,1)=sum(tn_index);
    
    fp_index=logical(predict_score>=threshold(i) & real_label==0);
    fp(i,1)=sum(fp_index);
    
    fn_index=logical(predict_score<threshold(i) & real_label==1);
    fn(i,1)=sum(fn_index);
end

sen=tp./(tp+fn);
precision=tp./(tp+fp);
recall=sen;
x=recall;
y=precision;
[x,index]=sort(x);
y=y(index,:);

area=0;
x(1,1)=0;
y(1,1)=1;
x(threshold_num+1,1)=1;
y(threshold_num+1,1)=0;
area=0.5*x(1)*(1+y(1));
for i=1:threshold_num
    area=area+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
end
plot(x,y)
end

function area=AUC(real_label,predict_score)
sorted_predict_score=unique(sort(predict_score));
score_num=size(sorted_predict_score,1);
threshold=sorted_predict_score(ceil(score_num*(1:999)/1000));

threshold=threshold';
threshold_num=length(threshold);
tn=zeros(threshold_num,1);
tp=zeros(threshold_num,1);
fn=zeros(threshold_num,1);
fp=zeros(threshold_num,1);
for i=1:threshold_num
    tp_index=logical(predict_score>=threshold(i) & real_label==1);
    tp(i,1)=sum(tp_index);
    
    tn_index=logical(predict_score<threshold(i) & real_label==0);
    tn(i,1)=sum(tn_index);
    
    fp_index=logical(predict_score>=threshold(i) & real_label==0);
    fp(i,1)=sum(fp_index);
    
    fn_index=logical(predict_score<threshold(i) & real_label==1);
    fn(i,1)=sum(fn_index);
end

sen=tp./(tp+fn);
spe=tn./(tn+fp);
y=sen;
x=1-spe;
[x,index]=sort(x);
y=y(index,:);
[y,index]=sort(y);
x=x(index,:);

area=0;
x(threshold_num+1,1)=1;
y(threshold_num+1,1)=1;
area=0.5*x(1)*y(1);
for i=1:threshold_num
    area=area+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
end
plot(x,y)
end

function W=GetLNSimilarity(feature_matrix,neighbor_num)
%线性邻居相似度的快速计算方法(W==similarity_matrix)
iteration_max=40;
mu=3;
X=feature_matrix;

%用欧氏距离计算邻居
row_num=size(X,1);
distance_matrix=pdist2(X,X,'euclidean');
e=ones(row_num,1);
distance_matrix=distance_matrix+diag(e*inf);
[~, si]=sort(distance_matrix,2,'ascend');
nearst_neighbor_matrix=zeros(row_num,row_num);
index=si(:,1:neighbor_num);
for i=1:row_num
    nearst_neighbor_matrix(i,index(i,:))=1;
end

%迭代求相似矩阵
C=nearst_neighbor_matrix;
rand('state',1);
W=rand(row_num,row_num);
W=(C.*W);
lamda=mu*e;
P=X*X'+lamda*e';
for i=1:iteration_max
    Q=(C.*W)*P;
    W=(C.*W).*P./Q;
    W(isnan(W))=0;
end
end