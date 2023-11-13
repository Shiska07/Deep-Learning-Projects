#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define MAX_PATH 500

void renameFilesInFolder(const char *folderPath, const char *extension)
{
    DIR *dir;
    struct dirent *entry;
    int count = 1;

    dir = opendir(folderPath);

    if (dir == NULL)
    {
        perror("Error opening directory");
        exit(EXIT_FAILURE);
    }

    while ((entry = readdir(dir)) != NULL)
    {
        if (entry->d_type == DT_REG)
        {
            char oldName[MAX_PATH];
            char newName[MAX_PATH];

            // Extract the original file extension
            const char *fileExtension = strrchr(entry->d_name, '.');
            if (fileExtension == NULL)
            {
                fprintf(stderr, "Error: File %s has no extension.\n", entry->d_name);
                exit(EXIT_FAILURE);
            }

            snprintf(oldName, sizeof(oldName), "%s/%s", folderPath, entry->d_name);
            snprintf(newName, sizeof(newName), "%s/%d.%s", folderPath, count, extension);

            if (rename(oldName, newName) != 0)
            {
                perror("Error renaming file");
                exit(EXIT_FAILURE);
            }

            count++;
        }
    }

    closedir(dir);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <folder_path> <file_extension>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *folderPath = argv[1];
    const char *extension = argv[2];

    renameFilesInFolder(folderPath, extension);

    printf("Files in %s renamed numerically with the extension .%s.\n", folderPath, extension);

    return 0;
}
