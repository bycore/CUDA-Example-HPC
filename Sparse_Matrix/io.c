#include <stdio.h>

int cpeek(FILE* file) {
    char ans = getc(file);
    ans = ungetc(ans, file);
    return ans;
}

void FEIL_ignore(FILE* file,size_t n, int delim) {
    while (n--) {
        const int c = getc(file);
        if (c == EOF)
            break;
        if (delim != EOF && delim == c) {
            break;
        }
    }
}
static void conv() {
    const char* filename = "demo.txt";
    FILE* file = fopen(filename, "r");
    while (cpeek(file) == '%') {
        FEIL_ignore(file,2048,'\n');
    }

    return;
}
int main() {
    conv();
}