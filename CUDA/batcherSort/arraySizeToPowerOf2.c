#include <stdio.h>
#include <stdlib.h>
#define TARGET_COUNT 2097152 // 2^21 

long count_numbers(FILE *fp) {
        long count = 0;
        float temp;
        while(fscanf(fp, "%f", &temp) == 1 ) {
                count++;
        }
        return count;
}

int main(int argc, char *argv[]){
        if (argc != 2) {
                printf("Use : %s <number_array.txt>\n", argv[0]);
                return -1;
        }

        const char *file_name = argv[1];
        FILE *fp = fopen(file_name, "r");
        if(!fp){
                printf("Error while opening the file for reading...\n");
                return -1;
        }
        
        long total_count = count_numbers(fp);
        printf("Total number count on original file: %ld\n", total_count);

        if(total_count < TARGET_COUNT){
                printf("File does't have enough numbers...\n");
                return -1;
        }

        rewind(fp);

        float *target_numbers = malloc(TARGET_COUNT * sizeof(float));

        for (long i = 0 ; i < TARGET_COUNT; ++i){
                if(fscanf(fp, "%f", &target_numbers[i]) != 1){
                        printf("Unexpected error while accessing the file...\n");
                        free(target_numbers);
                        fclose(fp);
                        return -1;
                }
        }

        fclose(fp);

        fp = fopen(file_name,"w");
        if(!fp){
                printf("Error while opening the file for writing...\n");
                free(target_numbers);
                return -1;
        }

        for(long i = 0 ; i < TARGET_COUNT; ++i){
                fprintf(fp,"%.6f\n",target_numbers[i]);
        }

        printf("File successfully truncated to %d numbers\n",TARGET_COUNT);

        free(target_numbers);
        fclose(fp);

        return 0;

}
