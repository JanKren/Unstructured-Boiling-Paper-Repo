!==============================================================================!
  subroutine User_Mod_End_Of_Compute_Vof(Vof, Sol)
!------------------------------------------------------------------------------!
!   This function is called at the end of Compute_Vof function.                !
!------------------------------------------------------------------------------!
  implicit none
!---------------------------------[Arguments]----------------------------------!
  type(Vof_Type),    target :: Vof
  type(Solver_Type), target :: Sol
  type(Field_Type),  pointer :: Flow
!-----------------------------------[Locals]-----------------------------------!
  type(Grid_Type),   pointer :: Grid
  type(Var_Type),    pointer :: fun
  type(Var_Type),    pointer :: u
  type(Var_Type),    pointer :: v
  type(Var_Type),    pointer :: w
  type(Var_Type),    pointer :: t
  type(Matrix_Type), pointer :: A
  integer                    :: c, d
  real                       :: x_ref, f_ref, t_ref, u_ref, v_ref, w_ref
!==============================================================================!

  ! Take aliases
  Flow => Vof  % pnt_flow
  Grid => Flow % pnt_grid
  fun  => Vof % fun
  A    => Sol % Nat % A
  u => Flow % u
  v => Flow % v
  w => Flow % w
  t => Flow % t

  do c = 1, Grid % n_cells

    ! Find relevant cell
    if( Math % Approx_Real(Grid % yc(c), 0.0) .and.  &
        Math % Approx_Real(Grid % zc(c), 0.0) ) then
      x_ref = Grid % xc(c)
      f_ref = fun % n(c)
      t_ref = t % n(c)
      u_ref = u % n(c)
      v_ref = v % n(c)
      w_ref = w % n(c)
      ! Browse through all other cells and homogenize the values
      do d = 1, Grid % n_cells
        if(d .ne. c) then
          if(Math % Approx_Real(Grid % xc(d), x_ref)) then
          fun % n(d) = f_ref
          t % n(c) = t_ref
          u % n(c) = u_ref
          v % n(c) = v_ref
          w % n(c) = w_ref
          end if
        end if
      end do
    end if
  end do

  end subroutine
