#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>   
#include <omp.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace py = pybind11;
py::dict run_simulation(py::array_t<double, py::array::c_style | py::array::forcecast> pos_array, 
                        py::array_t<double, py::array::c_style | py::array::forcecast> vel_array, 
                        int M, int N, double L, double m, double dt, 
                        double radius, int steps, int k_pairs, double target_ke) {
    
    auto pos = pos_array.mutable_unchecked<3>();
    auto vel = vel_array.mutable_unchecked<3>();

    std::vector<double> time_data(steps, 0.0);
    std::vector<double> ensemble_pressure_data(steps, 0.0);
    std::vector<double> ke_data(steps, 0.0);
    std::vector<double> mean_abs_px_data(steps, 0.0);
    std::vector<double> mean_abs_py_data(steps, 0.0);

    double diam_sq = (2 * radius) * (2 * radius);

    {
        py::gil_scoped_release release;

        for (int step = 0; step < steps; ++step) {
            time_data[step] = (step + 1) * dt;

            double total_P = 0.0;
            double total_KE = 0.0;
            double total_px = 0.0;
            double total_py = 0.0;

            #pragma omp parallel
            {
                std::mt19937 rng(42 + omp_get_thread_num() + step);
                std::uniform_int_distribution<int> dist(0, N - 1);

                double local_P = 0.0;
                double local_KE = 0.0;
                double local_px = 0.0;
                double local_py = 0.0;

                #pragma omp for
                for (int u = 0; u < M; ++u) {
                    double wall_impulse = 0.0;

                    for (int i = 0; i < N; ++i) {
                        double px = pos(u, i, 0) + vel(u, i, 0) * dt;
                        double py = pos(u, i, 1) + vel(u, i, 1) * dt;
                        double vx = vel(u, i, 0);
                        double vy = vel(u, i, 1);

                        if (px <= radius) { vx = std::abs(vx); wall_impulse += 2 * m * vx; px = radius; }
                        else if (px >= L - radius) { vx = -std::abs(vx); wall_impulse += 2 * m * std::abs(vx); px = L - radius; }

                        if (py <= radius) { vy = std::abs(vy); wall_impulse += 2 * m * vy; py = radius; }
                        else if (py >= L - radius) { vy = -std::abs(vy); wall_impulse += 2 * m * std::abs(vy); py = L - radius; }

                        pos(u, i, 0) = px; pos(u, i, 1) = py;
                        vel(u, i, 0) = vx; vel(u, i, 1) = vy;
                    }

                    local_P += wall_impulse / (dt * 4 * L);

                    for (int k = 0; k < k_pairs; ++k) {
                        int i = dist(rng);
                        int j = dist(rng);
                        if (i == j) j = (j + 1) % N;

                        double dx = pos(u, j, 0) - pos(u, i, 0);
                        double dy = pos(u, j, 1) - pos(u, i, 1);
                        double dist_sq = dx * dx + dy * dy;

                        if (dist_sq < diam_sq && dist_sq > 1e-10) {
                            double distance = std::sqrt(dist_sq);
                            double nx = dx / distance;
                            double ny = dy / distance;

                            double dvx = vel(u, j, 0) - vel(u, i, 0);
                            double dvy = vel(u, j, 1) - vel(u, i, 1);
                            double dot = dvx * nx + dvy * ny;

                            if (dot < 0) { 
                                vel(u, i, 0) += dot * nx;
                                vel(u, i, 1) += dot * ny;
                                vel(u, j, 0) -= dot * nx;
                                vel(u, j, 1) -= dot * ny;

                                double overlap = 2 * radius - distance;
                                pos(u, i, 0) -= overlap * nx * 0.5;
                                pos(u, i, 1) -= overlap * ny * 0.5;
                                pos(u, j, 0) += overlap * nx * 0.5;
                                pos(u, j, 1) += overlap * ny * 0.5;
                            }
                        }
                    }

                    double current_ke = 0.0;
                    for (int i = 0; i < N; ++i) {
                        current_ke += 0.5 * m * (vel(u, i, 0) * vel(u, i, 0) + vel(u, i, 1) * vel(u, i, 1));
                    }
                    
                    local_KE += current_ke;

                    double scale = std::sqrt(target_ke / std::max(current_ke, 1e-10));
                    double abs_px = 0.0;
                    double abs_py = 0.0;

                    for (int i = 0; i < N; ++i) {
                        vel(u, i, 0) *= scale;
                        vel(u, i, 1) *= scale;
                        abs_px += m * vel(u, i, 0);
                        abs_py += m * vel(u, i, 1);
                    }
                    
                    local_px += std::abs(abs_px);
                    local_py += std::abs(abs_py);
                }

                #pragma omp atomic
                total_P += local_P;
                #pragma omp atomic
                total_KE += local_KE;
                #pragma omp atomic
                total_px += local_px;
                #pragma omp atomic
                total_py += local_py;
            }

            ensemble_pressure_data[step] = total_P / M;
            ke_data[step] = total_KE / M;
            mean_abs_px_data[step] = total_px / M;
            mean_abs_py_data[step] = total_py / M;
        }
    } 

    
    py::dict results;
    results["time_data"] = py::cast(time_data);
    results["ensemble_pressure"] = py::cast(ensemble_pressure_data);
    results["ke_data"] = py::cast(ke_data);
    results["mean_abs_px"] = py::cast(mean_abs_px_data);
    results["mean_abs_py"] = py::cast(mean_abs_py_data);
    
    return results;
}

PYBIND11_MODULE(fast_ensemble, m) {
    m.def("run_simulation", &run_simulation, "Run the OpenMP accelerated gas ensemble");
}