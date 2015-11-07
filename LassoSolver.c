#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Note: This is the code for using random parallel successive convex approximation (RPSCA) for solving Lasso problem
// The paper can be found here:
/* http://arxiv.org/abs/1406.3665  */


// Solve lasso problem min 0.5*|b-Ax|_F^2 + lambda * |x|_1  
// Optimization variable is x
// x dimension n x 1
// b dimension m x 1
// A dimension m x n
// (c) Meisam Razaviyayn 

double distance(int n, double *x, double *xs);
// distance of two vectors

double normone(int n, double *x);
// ell_1 norm of a vector x

double Update(int m, int n, double *ai, double *b, double *x,int i, double tau, double lambda, double *Ax,double aiTai);
// Update rule of the RPSCA algorithm

int VecMatProd(int m, int n, double *A, double *x, double *Ax);
// Doing vector matrix multiplication


int main(int argc, char **argv)
{
    FILE  *fp;
    int i,j,k,m,n,blcklngth,size,my_rank,obj_counter,batchsize,index;
    double v , xiknew, objective,obj_loc,lambda,prev_obj,dist,dist_loc;
    double stepsize,startTime,endTime,ElapsedTime, frobA2, tau, norm2xs;
    MPI_Status status; // for MPI receive function
    int iter = 0;
    const int MAX_ITER  = 1000000;
    srand (time(NULL));
    objective = 0.0;
    fp = fopen("./Data/dimensions.txt", "r");
    fscanf (fp, "%d", &m);
    fscanf (fp, "%d", &n);
    lambda = 1;
    double *b_x;
    b_x = (double*) calloc(m,sizeof(double));
    double *Ax;
    Ax = (double*) calloc(m,sizeof(double));
    double *Ax_loc;
    Ax_loc = (double*)calloc(m,sizeof(double));


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size );
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    blcklngth = (int) ceil((n+0.0)/size);
    double n_total;
    n_total = n;  // Total number of columns in the matrix A
    if (my_rank < size-1)
        n = blcklngth;
    else
        n = (n - my_rank*blcklngth);  // Each core number of variables


/* Obtaining the matrix A as input */
    double *A;
    A = (double*) calloc(m*n,sizeof(double));
    fp = fopen("./Data/A.bin", "rb");
    if (!fp)
    {
    	printf("Unable to open file!");
    	return 1;
    }
    if (fp)
    {
    	for (j=0; j < (my_rank * m * blcklngth);j++)
    	{
    		fread((void*)(&v), sizeof(v), 1, fp);
	}
	for (j=0;j<n;j++)
	{
		for (i=0;i<m;i++)
		{
			fread((void*)(&v), sizeof(v), 1, fp);
			A[i+ m*j] = v;
	        }
		
	}
    }
    fclose(fp);

/* You can uncomment this part for printing the matrix A; please don't do it for large matrices though :) */
/*
    for (i=0;i<m;i++)
    {
        for (j=0;j<n;j++)
        {
            printf("%f ",A[i+m*j]);
        }
        printf("\n");
    }
*/

/* Calculate the Frobenius norm of A */
    frobA2=0;
    for (j=0;j<n;j++)
    {
        for (i=0;i<m;i++)
        {
            frobA2+= A[i+m*j]*A[i+m*j];
        }
    }
    MPI_Allreduce(&frobA2,&frobA2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
 //   tau = frobA2/(20 * n_total);


    /* Reading the vector b */
    double *b;
    b =(double*)calloc(m, sizeof(double));
    fp = fopen("./Data/b.bin", "rb");
    if (!fp)
    {
    	printf("Unable to open file!");
    	return 1;
    }
    if (fp)
    {
    	for (i=0;i<m;i++)
    	{
    		fread((void*)(&v), sizeof(v), 1, fp);
                b[i] = v;
    	}
    }
    fclose(fp);

/* You can uncomment this part for printing b */    
/*
    printf("b=");
    for (i=0;i<m;i++)
    {
        printf("%f ",b[i]);
        printf("\n");
    }

*/


/* Reading the actual solution obtained by Nesterov suggestion */ 
    // Also calculating its norm
    double *xs;
    xs = (double*) calloc(n,sizeof(double));
    fp = fopen("./Data/x.bin", "rb");
    if (!fp)
    {
    	printf("Unable to open file!");
	return 1;
    }
    if (fp)
    {
    	for (j=0; j < (my_rank * blcklngth);j++)
    	{
    		fread((void*)(&v), sizeof(v), 1, fp);
    	}
    	norm2xs = 0.0;
        for (j=0;j<n;j++)
        {
        	fread((void*)(&v), sizeof(v), 1, fp);
                xs[j] = v;
                norm2xs += v * v;
        }
    }
    fclose(fp);
    MPI_Allreduce(&norm2xs,&norm2xs,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);


// x is the optimization variable
    double *x;
    x = (double*)calloc(n,sizeof(double));

// aiTai is the variable for storing norm of each column of the matrix
    double *aiTai;
    aiTai = (double*)calloc(n,sizeof(double));
    for (i=0;i<n;i++)
    {
        for (j=0;j<m;j++)
        {
            aiTai[i] += A[m*i+j]*A[m*i+j];
        }
    }



/*
    if (my_rank==0)
    {
        printf("norm of xs = %f\n",norm2xs);
        printf("Process %d, x=",my_rank);
        for (j=0;j<m;j++)
        {
            printf("%f \n",Ax[j]);
            printf("\n");
        }
    }
    */

    iter = 0;
    // writing down the objective values for later plot
    fp = fopen("./Data/Objectives.m", "w");
    if (my_rank == 0)
    {
        fprintf(fp,"ObjValues = [ \n");
    }


    VecMatProd(m, n, A, x, Ax_loc);
    ElapsedTime = 0.0;
    tau = 0.5;
    stepsize = 0.9;
    batchsize = 1;
    i = 0 ;
    while (iter < MAX_ITER)
    {
        //stepsize = 0.99 - (1+0.0)/sqrt(sqrt(iter+4.1));
        startTime = MPI_Wtime();
        // i is the block chosen, ranging from 0 to n-1
        i = (i+batchsize)%(n);
        if(iter%100000 == 0)
        {
            VecMatProd(m, n, A, x, Ax_loc);
        }
        endTime = MPI_Wtime();
        ElapsedTime = ElapsedTime + endTime-startTime;
        MPI_Allreduce(Ax_loc,Ax,m,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        startTime = MPI_Wtime();
        for (k=0; k<batchsize;k++)
        {
            index = (i+k)%n;
            xiknew = Update(m,n,&A[m*index],b,x,index,tau,lambda,Ax, aiTai[index]); // updating the variable
            xiknew = x[index] + stepsize*(xiknew-x[index]);
            for (j=0;j<m;j++)
            {
                Ax_loc[j] = Ax_loc[j] + A[j+m*(index)]*(xiknew - x[index]);
            }
            x[index] = xiknew;
        }

/* different stepsize rules */
   //     stepsize = 0.01 + 0.1/(iter+1.0);
  //      stepsize = stepsize *(1-stepsize*0.00001);
 //       stepsize = 1;





 //           objective = obj (m,n,A,b,x,lambda,b_x); // uncomment if you want to stor objectives
        endTime = MPI_Wtime();
        ElapsedTime = ElapsedTime + endTime-startTime;

        if (iter%1000==0)
        {
            obj_loc = lambda*normone(n,x);
            dist_loc = distance(n,x,xs);
            prev_obj = objective;
            MPI_Allreduce(&obj_loc,&objective,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            MPI_Allreduce(&dist_loc,&dist,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            objective += 0.5 * distance(m,b,Ax);
            if ((my_rank==0)&&(iter%1000==0))
            {
                //objective = sqrt(objective/norm2xs);
                dist = sqrt(dist/norm2xs);
                endTime = MPI_Wtime();
                printf("iteration= %d, obj = %f , dist =%f, t=%f",iter,objective,dist,ElapsedTime);
                printf("\n");

                fprintf(fp,"%f   %f ; \n",ElapsedTime, dist);
            }

        }
        /*
        for (j=0;j<m;j++)
        {
            Ax_loc[j] = Ax_loc[j] + A[j+m*i]*(xinew - x[i]);
        }
        */

        iter += 1;


    }
    if (my_rank == 0)
    {
        fprintf(fp,"]; \n");
        fclose(fp);
    }
    MPI_Finalize();
    return 0;
}

int VecMatProd(int m, int n, double *A, double *x, double *Ax)
{
    int i,j;
    for (j=0;j<m;j++)
    {
        Ax [j] = 0.0;
        for (i=0;i<n;i++)
        {
            Ax[j] += A[j+m*i]*x[i];
        }
    }
    return 0;
}
double distance(int n, double *x, double *xs)
{
    int i;
    double temp,scale;
    scale = 0.0;
    for (i=0;i<n;i++)
    {
        temp = x[i]-xs[i];
        scale += temp*temp;
    }
    return scale;
}
double normone(int n, double *x)
{
    int k;
    double scale;
    scale = 0.0;
    for (k=0;k<n;k++)
    {
        scale += fabs(x[k]);
    }
    return scale;
}
double Update(int m, int n, double *ai, double *b, double *x,
            int i, double tau, double lambda, double *Ax, double aiTai)
{
    double xi,temp,tempaiaiT;
    int j;
    temp = 0.0;
    for (j=0;j<m;j++)
    {
        temp += ai[j] * (b[j]- Ax[j] + ai[j] * x[i]);
    }
    temp += tau * x[i];
    tempaiaiT = aiTai + tau;

    if (temp > lambda)
    {
        xi = (temp - lambda) / tempaiaiT;
    }
    else if (temp < -lambda)
    {
        xi = (temp + lambda) / tempaiaiT;
    }
    else
    {
        xi = 0.0;
    }
    return xi;
}
