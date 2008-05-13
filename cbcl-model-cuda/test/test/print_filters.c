#include<stdio.h>
int main()
{
  int i;
  for(i=0;i<4096;i++)
   printf("texture<float,2> tex_filt_%d;\n",i);

}
