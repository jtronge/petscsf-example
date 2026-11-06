// Little test of the PetscSF API.
#include <mpi.h>
#include <assert.h>
#include <petscsys.h>
#include <petscsf.h>

int main()
{
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(NULL, NULL, NULL, "test application for petsc-star forest"));

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(size == 2);

    // Choose some fake data.
    //
    // Assume our global matrix has 3 rows. local_ind contains the local partial
    // indices that we produce, and owned_ind is the disjoint set of indices
    // which we own and receive after a reduction.
    int local_ind[3];
    int local_count = 0;
    int owned_ind[3];
    int owned_count = 0;
    switch (rank) {
    case 0:
        local_ind[0] = 0;
        local_ind[1] = 2;
        local_count = 2;
        owned_ind[0] = 1;
        owned_count = 1;
        break;
    case 1:
        local_ind[0] = 0;
        local_ind[1] = 1;
        local_count = 2;
        owned_ind[0] = 0;
        owned_ind[1] = 2;
        owned_count = 2;
        break;
    default:
        break;
    }

    // This is the type for each row of the matrix.
    int width = 2;
    MPI_Datatype row_type;
    MPI_Type_contiguous(width, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    // See the documentation on PetscSF: https://petsc.org/main/manual/vec/#petscsf-an-alternative-to-low-level-mpi-calls-for-data-communication
    // Configure the root and leaf data to match the local/owned index abstraction from above.
    PetscInt nroots = owned_count;
    PetscInt nleaves = local_count;
    // roots should be the same length as local_count
    PetscSFNode roots[3];
    switch (rank) {
    case 0:
        roots[0].rank = 1;
        roots[0].index = 0;
        roots[1].rank = 1;
        roots[1].index = 1;
        break;
    case 1:
        roots[0].rank = 1;
        roots[0].index = 0;
        roots[1].rank = 0;
        roots[1].index = 0;
        break;
    default:
        break;
    }

    // Create the PetscSF object
    PetscSF sf;
    PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sf));
    PetscCall(PetscSFSetFromOptions(sf));
    PetscCall(PetscSFSetGraph(sf, nroots, nleaves, NULL, PETSC_COPY_VALUES, roots, PETSC_COPY_VALUES));
    PetscCall(PetscSFSetUp(sf));

    // Now set up some matrix data
    double* local_data = malloc(sizeof(double) * local_count * width);
    double* owned_data = malloc(sizeof(double) * owned_count * width);
    for (int i = 0; i < local_count; ++i) {
        for (int j = 0; j < width; ++j) {
            local_data[i * width + j] = 1;
        }
    }

    // Do the reduce-scatter
    PetscSFReduceBegin(sf, row_type, local_data, owned_data, MPIU_SUM);
    PetscSFReduceEnd(sf, row_type, local_data, owned_data, MPIU_SUM);

    // Check the data (this should just be summing the # of partials)
    printf("rank=%i,owned_data=[%g, %g, ...]\n", rank, owned_data[0], owned_data[1]);
    switch (rank) {
    case 0:
        assert(owned_data[0] == 1);
        assert(owned_data[1] == 1);
        break;
    case 1:
        assert(owned_data[0] == 2);
        assert(owned_data[1] == 2);
        assert(owned_data[2] == 1);
        assert(owned_data[3] == 1);
        break;
    default:
        break;
    }

    free(local_data);
    free(owned_data);

    PetscCall(PetscSFDestroy(&sf));

    MPI_Type_free(&row_type);
    PetscCall(PetscFinalize());
    return 0;
}
