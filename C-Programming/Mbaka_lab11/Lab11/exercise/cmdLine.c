/*---------------------------------------------------------------------------
 This program just echo's the command line parameters 
 student copy
---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {

   for (int i = 0; i < argc; i++) {
      printf("argv[%d]=%s\n", i, argv[i]);
   }

   return(0);
}
