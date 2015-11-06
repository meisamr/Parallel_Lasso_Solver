clear;
clc;
% This function generates data for LASSO according to Nestrov's paper 
% "Gradient Methods for Minimizing Composite Objective Function"



n=100000;                  % number of variables
N=1;                         % variable dimensions
m=1000;                     % observed data dimension
r_x=0.001;                % true data sparsity level
r_data=1;            % sparsity level for data matrix
rho=1;

   %% A is the data vector associated with each scalar variable
    A=zeros(m,n);

    x_true=zeros(n,N);
    m_star=ceil(r_x*n);
    
    B=zeros(m,n);
    for i=1:m
        for j=1:n
            r=rand(1);
            if r<r_data
                B(i,j)=rand(1)*2-1;
            else
               B(i,j)=0;
            end
        end
    end
    v=rand(m,1);
    y=v/norm(v);
    c=B'*y;
    
    [c ,index]=sort(abs(c),'descend');
    B=B(:,index);
    xi=rand(n,1);
    c=abs(c);
 
    for i=1:n
        if i<=m_star
            alpha=1/abs(c(i));
        elseif (i>m_star)&&(abs(c(i))<=0.1)
            alpha=1;
        else
            alpha=xi(i)/abs(c(i));
        end
        A(:,i)=alpha*B(:,i);
    end
    
      xi=rand(n,1)*rho/sqrt(m_star);
    for i=1:n
        if i<=m_star
            x_true(i)=xi(i)*sign(A(:,i)'*y);
        else
            x_true(i)=0;
        end
    end
    permut = randperm(n);
    A = A(:,permut);
    x_true = x_true(permut);
    b=y+A*x_true;
    
    OPT=0.5*norm(y)^2+norm(x_true,1);





fid = fopen('dimensions.txt', 'w');
fprintf(fid, '%d\n %d \n', m,n);
fclose(fid);




fid = fopen('A.bin', 'w');
fwrite(fid, A, 'double');
fclose(fid);

fid = fopen('b.bin', 'w');
fwrite(fid, b, 'double');
fclose(fid);

fid = fopen('x.bin', 'w');
fwrite(fid, x_true, 'double');
fclose(fid);
