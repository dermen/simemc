/*
This script is borrowed from:
    
    https://github.com/tl578/EMC-for-SMX/tree/master/make-quaternion

--------------------------------------------------------------------

Original header:

Generates quaternion sampling following the construction rule of the paper:
"A reconstruction algorithm for single-particle diffraction imaging experiments",
doi: 10.1103/PhysRevE.80.026705

compile:
gcc make-quaternion.c -O3 -lm -o quat.o

usage:
    ./quat.o (-ico) (-bin) num_div
    "-ico": reduce rotation subset by icosahedral symmetry (order = 60)
    "-bin": output binaray format

makes:
c-quaternion#.dat or c-ico-quaternion#.dat (ASCII)
or c-quaternion#.bin or c-ico-quaternion#.bin (binary)

Written in July 2015:
corrected the bug in sorting floating points;
calculated the weights with exact volumes of Voronoi cells, while
further improvement may be made if taking curvature into account

May 2016:
correct integer overflow problem as num_div becomes large

May 2017:
Allow binaray output format
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

/* golden mean */
#define tau 1.618033988749895
#define PI 3.141592653589793
#define num_vert 120
#define num_edge 720
#define num_face 1200
#define num_cell 600
/* number of nearest neighbors = 2*num_edge/num_vert */
#define nnn 12

void make_vertice() ;
void ver_even_permute( int ) ;
void make_edge() ;
void make_face() ;
void make_cell() ;
void make_map() ;
double weight( double*, double* ) ;
void quat_setup() ;
void refine_edge() ;
void refine_face() ;
void refine_cell() ;

struct q_point{
    int vec[4][2] ;
    double weight ;
} ;

void print_full_quat() ;
void free_mem() ;
int select_quat( struct q_point ) ;
int compare_ico_quat( struct q_point, struct q_point ) ;
int multi_0( int*, int* ) ;
int multi_1( int*, int* ) ;
void ico_transform( int[][2], int[][2] ) ;
void print_ico_quat() ;
void insert_sort( struct q_point*, struct q_point*, struct q_point ) ;

double (*quat)[5] ;
double vertices[num_vert][4] ;
int edges[num_edge][2] ;
int faces[num_face][3] ;
int cells[num_cell][4] ;
int nn_list[num_vert][nnn] ;
int edge2cell[num_edge][4] ;
int face2cell[num_face][4] ;
/* components a and b of the coordinate (a + b*tau)/(2*num_div) */
int vec_vertices[num_vert][4][2] ;
int num_edge_point, num_face_point, num_div, is_ico = 0, is_bin = 0 ;
long long num_cell_point ;
/* square of minimal distance between two vertices */
double min_dist2 ;
double f0, f1 ;
struct q_point *vertice_points, *edge_points, *face_points ;
struct q_point *cell_points, *ico_points, *cpy_ico_points ;

int main(int argc, char* argv[]){

    clock_t t = clock() ;

    if (argc == 2)
        num_div = atoi(argv[argc - 1]) ;
    else if (argc == 3){
        if (strstr(argv[1], "-ico")){
            is_ico = 1 ;
            printf("ico option is on!!\n") ;
        }
        else if (strstr(argv[1], "-bin")){
            is_bin = 1 ;
            printf("output format: binary\n") ;
        }
        num_div = atoi(argv[argc - 1]) ;
    }
    else if (argc == 4){
        if ((strstr(argv[1], "-ico") && strstr(argv[2], "-bin")) || \
            (strstr(argv[1], "-bin") && strstr(argv[2], "-ico"))){
            is_ico = 1 ;
            is_bin = 1 ;
            printf("ico option is on, output format: binary\n") ;
        }
        num_div = atoi(argv[argc - 1]) ;
    }
    else{
        printf("Wrong input format!!\n") ;
        return 1 ;
    }

    make_vertice() ;
    make_edge() ;
    make_face() ;
    make_cell() ;
    make_map() ;
    quat_setup() ;

    if (num_div > 1)
        refine_edge() ;
    if (num_div > 2)
        refine_face() ;
    if (num_div > 3)
        refine_cell() ;

    if (is_ico)
        print_ico_quat() ;
    else
        print_full_quat() ;

    free_mem() ;
    t = clock() - t ;
    printf("Computation time = %lf (s)\n", ((float)t)/CLOCKS_PER_SEC) ;
    return 0 ;
}


void make_vertice(){

    int h, i, j, k, idx = 0 ;

    /* 16 vertices */
    for (h = 0 ; h < 2 ; h++){
    for (i = 0 ; i < 2 ; i++){
    for (j = 0 ; j < 2 ; j++){
    for (k = 0 ; k < 2 ; k++){
        vertices[idx][0] = h - 0.5 ;
        vertices[idx][1] = i - 0.5 ;
        vertices[idx][2] = j - 0.5 ;
        vertices[idx][3] = k - 0.5 ;

        vec_vertices[idx][0][0] = (2*h - 1)*num_div ;
        vec_vertices[idx][1][0] = (2*i - 1)*num_div ;
        vec_vertices[idx][2][0] = (2*j - 1)*num_div ;
        vec_vertices[idx][3][0] = (2*k - 1)*num_div ;

        vec_vertices[idx][0][1] = 0 ;
        vec_vertices[idx][1][1] = 0 ;
        vec_vertices[idx][2][1] = 0 ;
        vec_vertices[idx][3][1] = 0 ;
        idx += 1 ;
    }
    }
    }
    }

    /* 8 vertices */
    for (h = 0 ; h < 2 ; h++){
    for (i = 0 ; i < 4 ; i++){
        for (j = 0 ; j < 4 ; j++){
            if (j == i){
                vertices[idx][j] = 2*h - 1.0 ;
                vec_vertices[idx][j][0] = (2*h - 1)*2*num_div ;
                vec_vertices[idx][j][1] = 0 ;
            }
            else{
                vertices[idx][j] = 0 ;
                vec_vertices[idx][j][0] = 0 ;
                vec_vertices[idx][j][1] = 0 ;
            }
        }
        idx += 1 ;
    }
    }

    /* the rest 96 vertices */
    ver_even_permute(idx) ;
}


void ver_even_permute( int idx ){

    int i, j, k, m, n ;

    /* even permutations */
    int perm_idx[12][4] = { {0, 1, 2, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {1, 2, 0, 3}, \
                    {1, 0, 3, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 3, 0, 1}, \
                    {2, 1, 3, 0}, {3, 1, 0, 2}, {3, 0, 2, 1}, {3, 2, 1, 0} } ;
    double vert[4] ;
    int vec_vert[4][2] ;

    for (i = 0 ; i < 2 ; i++){
    for (j = 0 ; j < 2 ; j++){
    for (k = 0 ; k < 2 ; k++){

        vert[0] = (2*i - 1)*tau/2. ;
        vert[1] = (2*j - 1)*0.5 ;
        vert[2] = (2*k - 1)/(2*tau) ;
        vert[3] = 0 ;

        vec_vert[0][0] = 0 ;
        vec_vert[0][1] = (2*i - 1)*num_div ;
        vec_vert[1][0] = (2*j - 1)*num_div ;
        vec_vert[1][1] = 0 ;
        vec_vert[2][0] = -(2*k - 1)*num_div ;
        vec_vert[2][1] = (2*k - 1)*num_div ;
        vec_vert[3][0] = 0 ;
        vec_vert[3][1] = 0 ;

        for (m = 0 ; m < 12 ; m++){
            for (n = 0 ; n < 4 ; n++){
                vertices[idx][n] = vert[perm_idx[m][n]] ;
                vec_vertices[idx][n][0] = vec_vert[perm_idx[m][n]][0] ;
                vec_vertices[idx][n][1] = vec_vert[perm_idx[m][n]][1] ;
            }
            idx += 1 ;
        }
    }
    }
    }
}


void make_edge(){

    double tmp, epsilon = 1.e-6 ;
    int i, j, k, edge_count = 0 ;
    int nn_count[num_vert] ;

    for (i = 0 ; i < 4 ; i++)
        min_dist2 += pow(vertices[0][i] - vertices[1][i], 2) ;

    for (i = 2 ; i < num_vert ; i++){
        tmp = 0 ;
        for (j = 0 ; j < 4 ; j++)
            tmp += pow(vertices[0][j] - vertices[i][j], 2) ;
        if (tmp < min_dist2)
            min_dist2 = tmp ;
    }

    /* offset by a small number to avoid the round-off error */
    min_dist2 += epsilon ;

    for (i = 0 ; i < num_vert ; i++)
        nn_count[i] = 0 ;

    for (i = 0 ; i < num_vert ; i++){
        for (j = i + 1 ; j < num_vert ; j++){
            tmp = 0 ;
            for (k = 0 ; k < 4 ; k++)
                tmp += pow(vertices[i][k] - vertices[j][k], 2) ;

            if (tmp < min_dist2){
                /* edges[*][0] < edges[*][1] */
                edges[edge_count][0] = i ;
                edges[edge_count][1] = j ;
                nn_list[i][nn_count[i]] = j ;
                nn_list[j][nn_count[j]] = i ;
                nn_count[i] += 1 ;
                nn_count[j] += 1 ;
                edge_count += 1 ;
            }
        }
    }
}


void make_face(){

    int i, j, k, idx ;
    int face_count = 0 ;
    double tmp ;

    for (i = 0 ; i < num_edge ; i++){
        for (j = 0 ; j < nnn ; j++){
            if (nn_list[edges[i][0]][j] <= edges[i][1])
                continue ;

            idx = nn_list[edges[i][0]][j] ;
            tmp = 0 ;
            for (k = 0 ; k < 4 ; k++)
                tmp += pow(vertices[idx][k] - vertices[edges[i][1]][k], 2) ;

            if (tmp < min_dist2){
                /* faces[*][0] < faces[*][1] < faces[*][2] */
                faces[face_count][0] = edges[i][0] ;
                faces[face_count][1] = edges[i][1];
                faces[face_count][2] = idx ;
                face_count += 1 ;
            }
        }
    }
}


void make_cell(){

    int i, j, k, idx ;
    int cell_count = 0 ;
    double tmp ;

    for (i = 0 ; i < num_face ; i++){
        for (j = 0 ; j < nnn ; j++){
            if (nn_list[faces[i][0]][j] <= faces[i][2])
                continue ;

            idx = nn_list[faces[i][0]][j] ;
            tmp = 0 ;
            for (k = 0 ; k < 4 ; k++)
                tmp += pow(vertices[idx][k] - vertices[faces[i][1]][k], 2) ;

            if (tmp > min_dist2)
                continue ;

            tmp = 0 ;
            for (k = 0 ; k < 4 ; k++)
                tmp += pow(vertices[idx][k] - vertices[faces[i][2]][k], 2) ;

            if (tmp > min_dist2)
                continue ;

            /* cells[*][0] < cells[*][1] < cells[*][2] < cells[*][3] */
            cells[cell_count][0] = faces[i][0] ;
            cells[cell_count][1] = faces[i][1] ;
            cells[cell_count][2] = faces[i][2] ;
            cells[cell_count][3] = idx ;
            cell_count += 1 ;
        }
    }
}


void make_map(){

    int i, j, k, m, idx ;
    double tmp ;

    /* face2cell */
    for (i = 0 ; i < num_face ; i++){
        for (j = 0 ; j < nnn ; j++){
            idx = nn_list[faces[i][0]][j] ;
            if (idx == faces[i][1] || idx == faces[i][2])
                continue ;

            tmp = 0 ;
            for (k = 0 ; k < 4 ; k++)
                tmp += pow(vertices[idx][k] - vertices[faces[i][1]][k], 2) ;

            if (tmp > min_dist2)
                continue ;

            tmp = 0 ;
            for (k = 0 ; k < 4 ; k++)
                tmp += pow(vertices[idx][k] - vertices[faces[i][2]][k], 2) ;

            if (tmp > min_dist2)
                continue ;

            face2cell[i][0] = faces[i][0] ;
            face2cell[i][1] = faces[i][1] ;
            face2cell[i][2] = faces[i][2] ;
            face2cell[i][3] = idx ;
            break ;
        }
    }

    /* edge2cell */
    for (i = 0 ; i < num_edge ; i++){
        for (j = 0 ; j < nnn ; j++){
            idx = nn_list[edges[i][0]][j] ;
            if (idx == edges[i][1])
                continue ;

            tmp = 0 ;
            for (k = 0 ; k < 4 ; k++)
                tmp += pow(vertices[idx][k] - vertices[edges[i][1]][k], 2) ;

            if (tmp > min_dist2)
                continue ;

            edge2cell[i][0] = edges[i][0] ;
            edge2cell[i][1] = edges[i][1] ;
            edge2cell[i][2] = idx ;

            for (k = j + 1 ; k < nnn ; k++){
                idx = nn_list[edges[i][0]][k] ;
                if (idx == edge2cell[i][1])
                    continue ;

                tmp = 0 ;
                for (m = 0 ; m < 4 ; m++)
                    tmp += pow(vertices[idx][m] - vertices[edge2cell[i][1]][m], 2) ;

                if (tmp > min_dist2)
                    continue ;

                tmp = 0 ;
                for (m = 0 ; m < 4 ; m++)
                    tmp += pow(vertices[idx][m] - vertices[edge2cell[i][2]][m], 2) ;

                if (tmp > min_dist2)
                    continue ;

                edge2cell[i][3] = idx ;
                break ;
            }
            break ;
        }
    }
}


double weight( double *v_q, double *v_c ){

    int i ;
    double w = 0, norm_q = 0, norm_c = 0 ;

    for (i = 0 ; i < 4 ; i++){
        norm_q += pow(v_q[i], 2) ;
        norm_c += pow(v_c[i], 2) ;
    }

    norm_q = sqrt(norm_q) ;
    norm_c = sqrt(norm_c) ;
    for (i = 0 ; i < 4 ; i++)
        w += v_q[i]*v_c[i] ;

    w /= pow(norm_q, 4)*norm_c ;
    return w ;
}


void quat_setup(){

    int i, j, k, m, visited_vert[num_vert] ;
    long long num_rot ;
    double v_q[4], v_c[4], w ;

    f0 = 5./6 ;
    f1 = 35./36 ;

    if (is_ico)
        num_rot = 10*(((long long) 5)*num_div*num_div*num_div + num_div) / (num_vert/2) ;
    else
        num_rot = 10*(((long long) 5)*num_div*num_div*num_div + num_div) ;

    quat = malloc(num_rot * sizeof(*quat)) ;
    vertice_points = malloc(num_vert * sizeof(struct q_point)) ;

    for (i = 0 ; i < num_vert ; i++)
        visited_vert[i] = 0 ;

    for (i = 0 ; i < num_cell ; i++){
        for (j = 0 ; j < 4 ; j++){

            if (visited_vert[cells[i][j]] == 1)
                continue ;

            visited_vert[cells[i][j]] = 1 ;
            for (k = 0 ; k < 4 ; k++){
                for (m = 0 ; m < 2 ; m++)
                    vertice_points[cells[i][j]].vec[k][m] = vec_vertices[cells[i][j]][k][m] ;
            }

            for (k = 0 ; k < 4 ; k++){
                v_c[k] = 0. ;
                for (m = 0 ; m < 4 ; m++)
                    v_c[k] += vertices[cells[i][m]][k] ;
                v_q[k] = vertices[cells[i][j]][k] ;
            }

            w = f0*weight(v_q, v_c) ;
            vertice_points[cells[i][j]].weight = w ;
        }
    }
}


void refine_edge(){

    int i, j, k ;
    double v_q[4], v_c[4], w ;
    int vec_d_v[4][2], edge_point_count = 0 ;

    num_edge_point = num_edge*(num_div - 1) ;
    edge_points = malloc(num_edge_point * sizeof(struct q_point)) ;
    for (i = 0 ; i < num_edge ; i++){
        for (j = 0 ; j < 4 ; j++){
            vec_d_v[j][0] = (vec_vertices[edges[i][1]][j][0] - vec_vertices[edges[i][0]][j][0]) / num_div ;
            vec_d_v[j][1] = (vec_vertices[edges[i][1]][j][1] - vec_vertices[edges[i][0]][j][1]) / num_div ;
        }

        for (j = 0 ; j < 4 ; j++){
            v_c[j] = 0. ;
            for (k = 0 ; k < 4 ; k++)
                v_c[j] += vertices[edge2cell[i][k]][j] ;
        }

        for (j = 1 ; j < num_div ; j++){
            for (k = 0 ; k < 4 ; k++){
                edge_points[edge_point_count].vec[k][0] = vec_vertices[edges[i][0]][k][0] + j*vec_d_v[k][0] ;
                edge_points[edge_point_count].vec[k][1] = vec_vertices[edges[i][0]][k][1] + j*vec_d_v[k][1] ;
                v_q[k] = (edge_points[edge_point_count].vec[k][0] + tau*edge_points[edge_point_count].vec[k][1]) / (2.0*num_div) ;
            }

            w = f1*weight(v_q, v_c) ;
            edge_points[edge_point_count].weight = w ;
            edge_point_count += 1 ;
        }
    }
}


void refine_face(){

    int i, j, k, m ;
    double v_q[4], v_c[4], w ;
    int vec_d_v1[4][2], vec_d_v2[4][2], face_point_count = 0 ;

    num_face_point = num_face*(num_div - 2)*(num_div - 1)/2 ;
    face_points = malloc(num_face_point * sizeof(struct q_point)) ;
    for (i = 0 ; i < num_face ; i++){
        for (j = 0 ; j < 4 ; j++){
            vec_d_v1[j][0] = (vec_vertices[faces[i][1]][j][0] - vec_vertices[faces[i][0]][j][0]) / num_div ;
            vec_d_v1[j][1] = (vec_vertices[faces[i][1]][j][1] - vec_vertices[faces[i][0]][j][1]) / num_div ;
            vec_d_v2[j][0] = (vec_vertices[faces[i][2]][j][0] - vec_vertices[faces[i][0]][j][0]) / num_div ;
            vec_d_v2[j][1] = (vec_vertices[faces[i][2]][j][1] - vec_vertices[faces[i][0]][j][1]) / num_div ;
        }

        for (j = 0 ; j < 4 ; j++){
            v_c[j] = 0. ;
            for (k = 0 ; k < 4 ; k++)
                v_c[j] += vertices[face2cell[i][k]][j] ;
        }

        for (j = 1 ; j < num_div - 1 ; j++){
            for (k = 1 ; k < num_div - j ; k++){
                for (m = 0 ; m < 4 ; m++){
                    face_points[face_point_count].vec[m][0] = vec_vertices[faces[i][0]][m][0] + j*vec_d_v1[m][0] + k*vec_d_v2[m][0] ;
                    face_points[face_point_count].vec[m][1] = vec_vertices[faces[i][0]][m][1] + j*vec_d_v1[m][1] + k*vec_d_v2[m][1] ;
                    v_q[m] = (face_points[face_point_count].vec[m][0] + tau*face_points[face_point_count].vec[m][1]) / (2.0*num_div) ;
                }

                w = weight(v_q, v_c) ;
                face_points[face_point_count].weight = w ;
                face_point_count += 1 ;
            }
        }
    }
}


void refine_cell(){

    int i, j, k, m, n ;
    double v_q[4], v_c[4], w ;
    int vec_d_v1[4][2], vec_d_v2[4][2], vec_d_v3[4][2] ;
    long long cell_point_count = 0 ;

    num_cell_point = ((long long) num_cell)/6*(num_div - 3)*(num_div - 2)*(num_div - 1) ;
    cell_points = malloc(num_cell_point * sizeof(struct q_point)) ;
    if (cell_points == NULL){
        printf("insufficient memory\n") ;
        exit(1) ;
    }

    for (i = 0 ; i < num_cell ; i++){
        for (j = 0 ; j < 4 ; j++){
            vec_d_v1[j][0] = (vec_vertices[cells[i][1]][j][0] - vec_vertices[cells[i][0]][j][0]) / num_div ;
            vec_d_v1[j][1] = (vec_vertices[cells[i][1]][j][1] - vec_vertices[cells[i][0]][j][1]) / num_div ;
            vec_d_v2[j][0] = (vec_vertices[cells[i][2]][j][0] - vec_vertices[cells[i][0]][j][0]) / num_div ;
            vec_d_v2[j][1] = (vec_vertices[cells[i][2]][j][1] - vec_vertices[cells[i][0]][j][1]) / num_div ;
            vec_d_v3[j][0] = (vec_vertices[cells[i][3]][j][0] - vec_vertices[cells[i][0]][j][0]) / num_div ;
            vec_d_v3[j][1] = (vec_vertices[cells[i][3]][j][1] - vec_vertices[cells[i][0]][j][1]) / num_div ;
        }

        for (j = 0 ; j < 4 ; j++){
            v_c[j] = 0. ;
            for (k = 0 ; k < 4 ; k++)
                v_c[j] += vertices[cells[i][k]][j] ;
        }

        for (j = 1 ; j < num_div - 2 ; j++){
            for (k = 1 ; k < num_div - 1 - j ; k++){
                for (m = 1 ; m < num_div - j - k ; m++){
                    for (n = 0 ; n < 4 ; n++){
                        cell_points[cell_point_count].vec[n][0] = vec_vertices[cells[i][0]][n][0] + j*vec_d_v1[n][0] + k*vec_d_v2[n][0] + m*vec_d_v3[n][0] ;
                        cell_points[cell_point_count].vec[n][1] = vec_vertices[cells[i][0]][n][1] + j*vec_d_v1[n][1] + k*vec_d_v2[n][1] + m*vec_d_v3[n][1] ;
                        v_q[n] = (cell_points[cell_point_count].vec[n][0] + tau*cell_points[cell_point_count].vec[n][1]) / (2.0*num_div) ;
                    }

                    w = weight(v_q, v_c) ;
                    cell_points[cell_point_count].weight = w ;
                    cell_point_count += 1 ;
                }
            }
        }
    }
}


void print_full_quat(){

    FILE *fp ;
    char fname[128] ;
    int i, flag, num_rot_cpy ;
    long long r, num_rot, ct ;
    double q_v[4], q_norm ;

    num_rot = (num_vert + num_edge_point + num_face_point + num_cell_point) / 2 ;
    if (num_rot != 10*(((long long) 5)*num_div*num_div*num_div + num_div)){
        printf("wrong num_rot!!\n") ;
        return ;
    }

    num_rot_cpy = num_rot ;
    if (num_rot_cpy != num_rot)
        printf("num_rot overflows int32!!\n") ;

    printf("num_rot = %lld\n", num_rot) ;
    if (is_bin == 1){
        sprintf(fname, "c-quaternion%d.bin", num_div) ;
        fp = fopen(fname, "wb") ;
        fwrite(&num_rot_cpy, sizeof(int), 1, fp) ;
    }
    else{
        sprintf(fname, "c-quaternion%d.dat", num_div) ;
        fp = fopen(fname, "w") ;
        fprintf(fp, "%d\n", num_rot_cpy) ;
    }

    /* select half of the quaternions on vertices */
    ct = 0 ;
    for (r = 0 ; r < num_vert ; r++){
        flag = select_quat(vertice_points[r]) ;
        if (flag != 1)
            continue ;

        q_norm = 0 ;
        for (i = 0 ; i < 4 ; i++){
            q_v[i] = (vertice_points[r].vec[i][0] + tau*vertice_points[r].vec[i][1]) / (2.0*num_div) ;
            q_norm += pow(q_v[i], 2) ;
        }

        q_norm = sqrt(q_norm) ;
        for (i = 0 ; i < 4 ; i++)
            q_v[i] /= q_norm ;

        if (is_bin == 1){
            fwrite(&q_v[0], sizeof(double), 1, fp) ;
            fwrite(&q_v[1], sizeof(double), 1, fp) ;
            fwrite(&q_v[2], sizeof(double), 1, fp) ;
            fwrite(&q_v[3], sizeof(double), 1, fp) ;
            fwrite(&vertice_points[r].weight, sizeof(double), 1, fp) ;
        }
        else{
            fprintf(fp, "%.12lf %.12lf %.12lf ", q_v[0], q_v[1], q_v[2]) ;
            fprintf(fp, "%.12lf %.12lf\n", q_v[3], vertice_points[r].weight) ;
        }
        ct += 1 ;
    }

    if (ct != num_vert / 2){
        printf("wrong number of quaternions on vertices!!\n") ;
        return ;
    }

    /* select half of the quaternions on edges */
    ct = 0 ;
    for (r = 0 ; r < num_edge_point ; r++){
        flag = select_quat(edge_points[r]) ;
        if (flag != 1)
            continue ;

        q_norm = 0 ;
        for (i = 0 ; i < 4 ; i++){
            q_v[i] = (edge_points[r].vec[i][0] + tau*edge_points[r].vec[i][1]) / (2.0*num_div) ;
            q_norm += pow(q_v[i], 2) ;
        }

        q_norm = sqrt(q_norm) ;
        for (i = 0 ; i < 4 ; i++)
            q_v[i] /= q_norm ;

        if (is_bin == 1){
            fwrite(&q_v[0], sizeof(double), 1, fp) ;
            fwrite(&q_v[1], sizeof(double), 1, fp) ;
            fwrite(&q_v[2], sizeof(double), 1, fp) ;
            fwrite(&q_v[3], sizeof(double), 1, fp) ;
            fwrite(&edge_points[r].weight, sizeof(double), 1, fp) ;
        }
        else{
            fprintf(fp, "%.12lf %.12lf %.12lf ", q_v[0], q_v[1], q_v[2]) ;
            fprintf(fp, "%.12lf %.12lf\n", q_v[3], edge_points[r].weight) ;
        }
        ct += 1 ;
    }

    if (ct != num_edge_point / 2){
        printf("wrong number of quaternions on edges!!\n") ;
        return ;
    }

    /* select half of the quaternions on faces */
    ct = 0 ;
    for (r = 0 ; r < num_face_point ; r++){
        flag = select_quat(face_points[r]) ;
        if (flag != 1)
            continue ;

        q_norm = 0 ;
        for (i = 0 ; i < 4 ; i++){
            q_v[i] = (face_points[r].vec[i][0] + tau*face_points[r].vec[i][1]) / (2.0*num_div) ;
            q_norm += pow(q_v[i], 2) ;
        }

        q_norm = sqrt(q_norm) ;
        for (i = 0 ; i < 4 ; i++)
            q_v[i] /= q_norm ;

        if (is_bin == 1){
            fwrite(&q_v[0], sizeof(double), 1, fp) ;
            fwrite(&q_v[1], sizeof(double), 1, fp) ;
            fwrite(&q_v[2], sizeof(double), 1, fp) ;
            fwrite(&q_v[3], sizeof(double), 1, fp) ;
            fwrite(&face_points[r].weight, sizeof(double), 1, fp) ;
        }
        else{
            fprintf(fp, "%.12lf %.12lf %.12lf ", q_v[0], q_v[1], q_v[2]) ;
            fprintf(fp, "%.12lf %.12lf\n", q_v[3], face_points[r].weight) ;
        }
        ct += 1 ;
    }

    if (ct != num_face_point / 2){
        printf("wrong number of quaternions on faces!!\n") ;
        return ;
    }

    /* select half of the quaternions on cells */
    ct = 0 ;
    for (r = 0 ; r < num_cell_point ; r++){
        flag = select_quat(cell_points[r]) ;
        if (flag != 1)
            continue ;

        q_norm = 0 ;
        for (i = 0 ; i < 4 ; i++){
            q_v[i] = (cell_points[r].vec[i][0] + tau*cell_points[r].vec[i][1]) / (2.0*num_div) ;
            q_norm += pow(q_v[i], 2) ;
        }

        q_norm = sqrt(q_norm) ;
        for (i = 0 ; i < 4 ; i++)
            q_v[i] /= q_norm ;

        if (is_bin == 1){
            fwrite(&q_v[0], sizeof(double), 1, fp) ;
            fwrite(&q_v[1], sizeof(double), 1, fp) ;
            fwrite(&q_v[2], sizeof(double), 1, fp) ;
            fwrite(&q_v[3], sizeof(double), 1, fp) ;
            fwrite(&cell_points[r].weight, sizeof(double), 1, fp) ;
        }
        else{
            fprintf(fp, "%.12lf %.12lf %.12lf ", q_v[0], q_v[1], q_v[2]) ;
            fprintf(fp, "%.12lf %.12lf\n", q_v[3], cell_points[r].weight) ;
        }
        ct += 1 ;
    }

    if (ct != num_cell_point / 2){
        printf("wrong number of quaternions on cells!!\n") ;
        return ;
    }
    fclose(fp) ;
}


int multi_0( int q[2], int p[2] ){
    return q[0]*p[0] + q[1]*p[1] ;
}


int multi_1( int q[2], int p[2] ){
    return q[0]*p[1] + q[1]*p[0] + q[1]*p[1] ;
}


void ico_transform( int q[][2], int transformed[][2] ){

    int i, j, k, p[4][2], r[4][2], flag ;

    for (i = 0 ; i < num_vert ; i++){
        for (j = 0 ; j < 4 ; j++){
            for (k = 0 ; k < 2 ; k++)
                p[j][k] = vec_vertices[i][j][k] ;
        }

        /* quaternion transform rule:
          r0 = q0*p0 - (q1*p1 + q2*p2 + q3*p3) ;
          r1 = q0*p1 + p0*q1 + (q2*p3 - q3*p2) ;
          r2 = q0*p2 + p0*q2 + (q3*p1 - q1*p3) ;
          r3 = q0*p3 + p0*q3 + (q1*p2 - q2*p1) ; */

        r[0][0] = multi_0(q[0], p[0]) - ( multi_0(q[1], p[1]) + multi_0(q[2], p[2]) + multi_0(q[3], p[3]) ) ;
        r[0][1] = multi_1(q[0], p[0]) - ( multi_1(q[1], p[1]) + multi_1(q[2], p[2]) + multi_1(q[3], p[3]) ) ;

        r[1][0] = multi_0(q[0], p[1]) + multi_0(q[1], p[0]) + ( multi_0(q[2], p[3]) - multi_0(q[3], p[2]) ) ;
        r[1][1] = multi_1(q[0], p[1]) + multi_1(q[1], p[0]) + ( multi_1(q[2], p[3]) - multi_1(q[3], p[2]) ) ;

        r[2][0] = multi_0(q[0], p[2]) + multi_0(q[2], p[0]) + ( multi_0(q[3], p[1]) - multi_0(q[1], p[3]) ) ;
        r[2][1] = multi_1(q[0], p[2]) + multi_1(q[2], p[0]) + ( multi_1(q[3], p[1]) - multi_1(q[1], p[3]) ) ;

        r[3][0] = multi_0(q[0], p[3]) + multi_0(q[3], p[0]) + ( multi_0(q[1], p[2]) - multi_0(q[2], p[1]) ) ;
        r[3][1] = multi_1(q[0], p[3]) + multi_1(q[3], p[0]) + ( multi_1(q[1], p[2]) - multi_1(q[2], p[1]) ) ;

        for (j = 0 ; j < 4 ; j++){
            for (k = 0 ; k < 2 ; k++){
                if (r[j][k] % (2*num_div) != 0)
                    printf("mod (2*num_div) != 0\n") ;
                r[j][k] /= (2*num_div) ;
            }
        }

        if (i == 0){
            for (j = 0 ; j < 4 ; j++){
                for (k = 0 ; k < 2 ; k++)
                    transformed[j][k] = r[j][k] ;
            }
        }
        else{
            flag = 0 ;
            for (j = 0 ; j < 4 ; j++){
                for (k = 0 ; k < 2 ; k++){
                    if (r[j][k] < transformed[j][k]){
                        flag = 1 ;
                        break ;
                    }
                    else if (r[j][k] > transformed[j][k]){
                        flag = -1 ;
                        break ;
                    }
                }
                if (flag != 0)
                    break ;
            }
            if (flag == 1){
                for (j = 0 ; j < 4 ; j++){
                    for (k = 0 ; k < 2 ; k++)
                        transformed[j][k] = r[j][k] ;
                }
            }
        }
    }
}


void print_ico_quat(){

    FILE *fp ;
    char fname[128] ;
    int i, j, flag, num_rot_cpy, a0[4][2], b0[4][2] ;
    long long r, num_rot, ct, rot_ct = 0 ;
    double q_v[4], q_norm ;
    struct q_point r_p ;

    num_rot = (num_vert + num_edge_point + num_face_point + num_cell_point) / num_vert ;
    if (num_rot != 10*(((long long) 5)*num_div*num_div*num_div + num_div) / (num_vert/2)){
        printf("wrong num_rot!!\n") ;
        return ;
    }

    num_rot_cpy = num_rot ;
    if (num_rot_cpy != num_rot)
        printf("num_rot overflows int32!!\n") ;

    ico_points = malloc(num_rot * sizeof(struct q_point)) ;
    cpy_ico_points = malloc(num_rot * sizeof(struct q_point)) ;

    /* initialize */
    for (r = 0 ; r < num_rot ; r++){
        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2; j++){
                ico_points[r].vec[i][j] = 0 ;
                cpy_ico_points[r].vec[i][j] = 0 ;
            }
        }
        ico_points[r].weight = -1 ;
        cpy_ico_points[r].weight = -1 ;
    }

    /* select one quaternion on vertices */
    for (i = 0 ; i < 4 ; i++){
        for (j = 0 ; j < 2; j++)
            r_p.vec[i][j] = vertice_points[0].vec[i][j] ;
    }

    r_p.weight = vertice_points[0].weight ;
    for (r = 1 ; r < num_vert ; r++){
        flag = compare_ico_quat(vertice_points[r], r_p) ;
        if (flag == 1){
            for (i = 0 ; i < 4 ; i++){
                for (j = 0 ; j < 2; j++)
                    r_p.vec[i][j] = vertice_points[r].vec[i][j] ;
            }
            r_p.weight = vertice_points[r].weight ;
        }
    }

    for (i = 0 ; i < 4 ; i++){
        for (j = 0 ; j < 2; j++){
            ico_points[rot_ct].vec[i][j] = r_p.vec[i][j] ;
            cpy_ico_points[rot_ct].vec[i][j] = r_p.vec[i][j] ;
        }
    }

    ico_points[rot_ct].weight = r_p.weight ;
    cpy_ico_points[rot_ct].weight = r_p.weight ;
    rot_ct += 1 ;

    /* select quaternions on edges */
    for (r = 0 ; r < num_edge_point ; r++){
        flag = select_quat(edge_points[r]) ;
        if (flag != 1)
            continue ;

        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2; j++)
                a0[i][j] = edge_points[r].vec[i][j] ;
        }

        ico_transform(a0, b0) ;
        r_p.weight = edge_points[r].weight ;
        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2; j++)
                r_p.vec[i][j] = b0[i][j] ;
        }

        insert_sort(ico_points, cpy_ico_points, r_p) ;
    }

    ct = 0 ;
    for (r = 0 ; r < num_rot ; r++){
        if (ico_points[r].weight > 0)
            ct += 1 ;
    }

    if (ct - rot_ct != num_edge_point / num_vert){
        printf("wrong number of quaternions on edges!!\n") ;
        return ;
    }

    rot_ct = ct ;

    /* select quaternions on faces */
    ct = 0 ;
    for (r = 0 ; r < num_face_point ; r++){
        flag = select_quat(face_points[r]) ;
        if (flag != 1)
            continue ;

        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2; j++)
                a0[i][j] = face_points[r].vec[i][j] ;
        }

        ico_transform(a0, b0) ;
        r_p.weight = face_points[r].weight ;
        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2; j++)
                r_p.vec[i][j] = b0[i][j] ;
        }

        insert_sort(ico_points, cpy_ico_points, r_p) ;
    }

    ct = 0 ;
    for (r = 0 ; r < num_rot ; r++){
        if (ico_points[r].weight > 0)
            ct += 1 ;
    }

    if (ct - rot_ct != num_face_point / num_vert){
        printf("wrong number of quaternions on faces!!\n") ;
        return ;
    }

    rot_ct = ct ;

    /* select quaternions on cells */
    ct = 0 ;
    for (r = 0 ; r < num_cell_point ; r++){
        flag = select_quat(cell_points[r]) ;
        if (flag != 1)
            continue ;

        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2; j++)
                a0[i][j] = cell_points[r].vec[i][j] ;
        }

        ico_transform(a0, b0) ;
        r_p.weight = cell_points[r].weight ;
        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2; j++)
                r_p.vec[i][j] = b0[i][j] ;
        }

        insert_sort(ico_points, cpy_ico_points, r_p) ;
    }

    ct = 0 ;
    for (r = 0 ; r < num_rot ; r++){
        if (ico_points[r].weight > 0)
            ct += 1 ;
    }

    if (ct - rot_ct != num_cell_point / num_vert){
        printf("wrong number of quaternions on cells!!\n") ;
        return ;
    }

    rot_ct = ct ;
    printf("num_rot = %lld\n", ct) ;

    if (is_bin == 1){
        sprintf(fname, "c-ico-quaternion%d.bin", num_div) ;
        fp = fopen(fname, "wb") ;
        fwrite(&num_rot_cpy, sizeof(int), 1, fp) ;
    }
    else{
        sprintf(fname, "c-ico-quaternion%d.dat", num_div) ;
        fp = fopen(fname, "w") ;
        fprintf(fp, "%d\n", num_rot_cpy) ;
    }

    for (r = 0 ; r < num_rot ; r++){
        q_norm = 0 ;
        for (i = 0 ; i < 4 ; i++){
            q_v[i] = (ico_points[r].vec[i][0] + tau*ico_points[r].vec[i][1]) / (2.0*num_div) ;
            q_norm += pow(q_v[i], 2) ;
        }

        q_norm = sqrt(q_norm) ;
        for (i = 0 ; i < 4 ; i++)
            q_v[i] /= q_norm ;

        if (is_bin == 1){
            fwrite(&q_v[0], sizeof(double), 1, fp) ;
            fwrite(&q_v[1], sizeof(double), 1, fp) ;
            fwrite(&q_v[2], sizeof(double), 1, fp) ;
            fwrite(&q_v[3], sizeof(double), 1, fp) ;
            fwrite(&ico_points[r].weight, sizeof(double), 1, fp) ;
        }
        else{
            fprintf(fp, "%.12lf %.12lf %.12lf ", q_v[0], q_v[1], q_v[2]) ;
            fprintf(fp, "%.12lf %.12lf\n", q_v[3], ico_points[r].weight) ;
        }
    }

    fclose(fp) ;
    free(ico_points) ;
    free(cpy_ico_points) ;
}


void insert_sort( struct q_point *ico_points, struct q_point *cpy_ico_points, struct q_point r_p ){

    int i, j, flag = 0 ;
    long long r, num_rot, num_element, iBegin = 0, iEnd = 0, iMiddle = 0 ;
    num_rot = 10*(((long long) 5)*num_div*num_div*num_div + num_div) / (num_vert/2) ;

    /* iBegin: inclusive, iEnd: exclusive */
    iBegin = 0 ;
    for (r = 0 ; r < num_rot ; r++){
        if (ico_points[r].weight > 0)
            iEnd = r ;
        else
            break ;
    }

    iEnd += 1 ;
    num_element = iEnd ;
    flag = compare_ico_quat(r_p, ico_points[iBegin]) ;

    if (flag == 0)
        return ;
    else if (flag == 1){
        if (num_element >= num_rot){
            printf("Too many quaternions!!\n") ;
            return ;
        }

        /* insert r_p to the position before ico_points[iBegin] */
        for (r = 0 ; r < iEnd ; r++){
            for (i = 0 ; i < 4 ; i++){
                for (j = 0 ; j < 2 ; j++)
                    cpy_ico_points[r+1].vec[i][j] = ico_points[r].vec[i][j] ;
            }
            cpy_ico_points[r+1].weight = ico_points[r].weight ;
        }

        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2 ; j++)
                cpy_ico_points[iBegin].vec[i][j] = r_p.vec[i][j] ;
        }
        cpy_ico_points[iBegin].weight = r_p.weight ;

        for (r = 0 ; r < iEnd + 1 ; r++){
            for (i = 0 ; i < 4 ; i++){
                for (j = 0 ; j < 2 ; j++)
                    ico_points[r].vec[i][j] = cpy_ico_points[r].vec[i][j] ;
            }
            ico_points[r].weight = cpy_ico_points[r].weight ;
        }

        return ;
    }

    flag = compare_ico_quat(ico_points[iEnd-1], r_p) ;

    if (flag == 0)
        return ;
    else if (flag == 1){

        if (num_element >= num_rot){
            printf("Too many quaternions!!\n") ;
            return ;
        }

        /* insert r_p to ico_points[iEnd] */
        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2 ; j++){
                ico_points[iEnd].vec[i][j] = r_p.vec[i][j] ;
                cpy_ico_points[iEnd].vec[i][j] = r_p.vec[i][j] ;
            }
        }
        ico_points[iEnd].weight = r_p.weight ;
        cpy_ico_points[iEnd].weight = r_p.weight ;

        return ;
    }

    /* r_p rests between ico_points[iBegin] and ico_points[iEnd-1] */
    while (iEnd - iBegin != 1){
        iMiddle = (iBegin + iEnd) / 2 ;
        flag = compare_ico_quat(r_p, ico_points[iMiddle]) ;
        /* r_p is identical to ico_points[iMiddle] */
        if (flag == 0)
            return ;
        else if (flag == 1)
            iEnd = iMiddle ;
        else
            iBegin = iMiddle ;
    }

    if (flag == 1){
        if (num_element >= num_rot){
            printf("Too many quaternions!!\n") ;
            return ;
        }

        /* r_p precedes ico_points[iMiddle] */
        for (r = iMiddle ; r < num_element ; r++){
            for (i = 0 ; i < 4 ; i++){
                for (j = 0 ; j < 2 ; j++)
                    cpy_ico_points[r+1].vec[i][j] = ico_points[r].vec[i][j] ;
            }
            cpy_ico_points[r+1].weight = ico_points[r].weight ;
        }

        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2 ; j++)
                cpy_ico_points[iMiddle].vec[i][j] = r_p.vec[i][j] ;
        }
        cpy_ico_points[iMiddle].weight = r_p.weight ;

        for (r = iMiddle ; r < num_element + 1 ; r++){
            for (i = 0 ; i < 4; i++){
                for (j = 0 ; j < 2 ; j++)
                    ico_points[r].vec[i][j] = cpy_ico_points[r].vec[i][j] ;
            }
            ico_points[r].weight = cpy_ico_points[r].weight ;
        }
    }
    else{
        if (num_element >= num_rot){
            printf("Too many quaternions!!\n") ;
            return ;
        }

        /* r_p follows ico_points[iMiddle] */
        for (r = iMiddle + 1 ; r < num_element ; r++){
            for (i = 0 ; i < 4 ; i++){
                for (j = 0 ; j < 2 ; j++)
                    cpy_ico_points[r+1].vec[i][j] = ico_points[r].vec[i][j] ;
            }
            cpy_ico_points[r+1].weight = ico_points[r].weight ;
        }

        for (i = 0 ; i < 4 ; i++){
            for (j = 0 ; j < 2 ; j++)
                cpy_ico_points[iMiddle + 1].vec[i][j] = r_p.vec[i][j] ;
        }
        cpy_ico_points[iMiddle + 1].weight = r_p.weight ;

        for (r = iMiddle + 1 ; r < num_element + 1 ; r++){
            for (i = 0 ; i < 4; i++){
                for (j = 0 ; j < 2 ; j++)
                    ico_points[r].vec[i][j] = cpy_ico_points[r].vec[i][j] ;
            }
            ico_points[r].weight = cpy_ico_points[r].weight ;
        }
    }
}


int select_quat( struct q_point q_p ){

    int i, j, flag = 0 ;

    for (i = 0 ; i < 4 ; i++){
        for (j = 0 ; j < 2 ; j++){
            if (q_p.vec[i][j] > 0){
                flag = 1 ;
                break ;
            }
            else if (q_p.vec[i][j] < 0){
                flag = -1 ;
                break ;
            }
        }
        if (flag != 0)
            break ;
    }

    return flag ;
}


int compare_ico_quat( struct q_point r_p, struct q_point q_p ){

    /* return 1 if r_p precedes q_p, -1 if r_p follows q_p, 0 if r_p and q_p are the same */
    int i, j, flag = 0 ;

    for (i = 0 ; i < 4; i++){
        for (j = 0 ; j < 2 ; j++){
            if (r_p.vec[i][j] < q_p.vec[i][j]){
                flag = 1 ;
                break ;
            }
            else if (r_p.vec[i][j] > q_p.vec[i][j]){
                flag = -1 ;
                break ;
            }
        }
        if (flag != 0)
            break ;
    }

    return flag ;
}


void free_mem(){

    free(quat) ;
    free(vertice_points) ;
    if (num_div > 1)
        free(edge_points) ;
    if (num_div > 2)
        free(face_points) ;
    if (num_div > 3)
        free(cell_points) ;
}
