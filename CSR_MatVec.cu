#include<stdio.h>
#include<stdlib.h>

void read_matrix(int **r_ptr, int** c_ind,float** v, char*fname,int* r_count,int* v_count);

int main(){
    int* row_ptr;
    int* col_ind;
    float* value;
    int r_count, v_count;
    char* fname="" ;

    read_matrix(&row_ptr,&col_ind,&value,fname,&r_count,&v_count);
}

void read_matrix(int **r_ptr, int** c_ind,float** v, char*fname,int* r_count,int* v_count){
    FILE* file;
    if(NULL==(file = fopen(fname,"r+"))){
        fprintf(stderr,"Cannot open output file.\n");
        return ;
    }
    int colum_count,row_count,values_count;
    fscanf(file,"%d%d%d\n",&row_count,&colum_count,&values_count);
    *r_count = row_count;
	*v_count = values_count;
    int *row_ptr =(int*) malloc((row_count+1) * sizeof(int));
	int *col_ind =(int*) malloc(values_count * sizeof(int));
    for(int i=0; i<values_count; i++){
		col_ind[i] = -1;
	}
}