#include <stdio.h>
#define SIZE 1024
double a[SIZE][SIZE];
int i;
double  b[SIZE][SIZE];
void main( int argc, char **argv )
{
  int j,k;
  for( i=0; i < SIZE; i++ )
  { 
    for( j = 0; j < SIZE; j++ )
    {
       b[i][j] = 20.19;
    }
  }
  for( j = 0; j < SIZE; j++ )
  {
    for( i = 0; i < SIZE; i++ )
    {
      a[i][j] = 0;
      for( k = 0; k < SIZE; k++ )        
      a[i][j] += b[i][k]+b[j][k];
    }
  }
}
