A = sparse(gallery('poisson', 25)); % poisson matrix on 25x25 regular grid
save("test_matrix.mat", 'A')

A(1,2) = A(1,2) + 1e-14;
save("test_matrix_perturbed.mat", 'A')