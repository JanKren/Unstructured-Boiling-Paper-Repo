!==============================================================================!
  subroutine User_Mod_End_Of_Time_Step(Flow, Turb, Vof, Swarm,  &
                                       n_stat_t, n_stat_p)
!------------------------------------------------------------------------------!
!   Global mass budget, in the form used by Sato & Nicheno (JCP 2013, Fig. 8).  !
!                                                                              !
!   Writes `mass-balance.dat`, one row per time step:                          !
!                                                                              !
!       t      M_d      M_s      M_d/M_0      M_s/M_0 - 1                       !
!                                                                              !
!       M_d = sum_cells rho V_cell                     mass inside the domain   !
!       M_s = M_d + sum_t sum_outlet rho u.S dt        mass allowing for egress !
!                                                                              !
!   M_d falls as fluid leaves through the outflow; M_s must hold at M_0, and    !
!   (M_s/M_0 - 1) is the conservation error.                                   !
!                                                                              !
!   WHY IT CANNOT BE POST-PROCESSED.  M_d alone follows from the vapour volume  !
!   already in `bench-data.dat`.  The egress term is recorded nowhere, and      !
!   inferring it as M_0 - M_d would make M_s identically M_0 -- a flat line     !
!   produced by algebra, not by conservation.  The content of the check is that !
!   the two halves are measured INDEPENDENTLY and still sum to M_0.            !
!                                                                              !
!   Drop in as the case's End_Of_Time_Step.f90 for cases that have none         !
!   (Stefan, Sucking); for Scriven merge the marked block into the existing     !
!   hook so that `bench-data.dat` keeps being written.                          !
!                                                                              !
!   Loop forms are the explicit ones used by the paper's own cases rather than  !
!   the Browse.h90 macros, which not every case template includes.              !
!------------------------------------------------------------------------------!
  implicit none
!---------------------------------[Arguments]----------------------------------!
  type(Field_Type), target :: Flow
  type(Turb_Type),  target :: Turb
  type(Vof_Type),   target :: Vof
  type(Swarm_Type), target :: Swarm
  integer                  :: n_stat_t  ! 1st t.s. statistics turbulence
  integer                  :: n_stat_p  ! 1st t.s. statistics particles
!-----------------------------------[Locals]-----------------------------------!
  type(Grid_Type), pointer :: Grid
  integer                  :: c, c1, c2, s, fu
  real                     :: m_dom, out_rate, m_s
  real,               save :: out_cum  = 0.0    ! running egress [kg]
  real,               save :: out_prev = 0.0    ! previous step's rate
  real,               save :: m_zero   = 0.0    ! M_0
  logical,            save :: first    = .true.
!==============================================================================!

  Grid => Flow % pnt_grid

  !---------------------------------------!
  !   Mass currently inside the domain    !
  !---------------------------------------!
  m_dom = 0.0
  do c = 1, Grid % n_cells - Grid % Comm % n_buff_cells
    m_dom = m_dom + Flow % density(c) * Grid % vol(c)
  end do
  call Global % Sum_Real(m_dom)

  !------------------------------------------------------!
  !   Mass leaving through the outflow during this step  !
  !   v_flux is the volume flux, positive out of c1      !
  !------------------------------------------------------!
  out_rate = 0.0
  do s = 1, Grid % n_faces
    c1 = Grid % faces_c(1,s)
    c2 = Grid % faces_c(2,s)
    if(c2 < 0) then                       ! a boundary face
      if(Grid % Bnd_Cond_Type(c2) .eq. OUTFLOW   .or.  &
         Grid % Bnd_Cond_Type(c2) .eq. PRESSURE  .or.  &
         Grid % Bnd_Cond_Type(c2) .eq. CONVECT) then
        ! density of the interior cell: the boundary value is not always updated
        out_rate = out_rate + Flow % density(c1) * Flow % v_flux % n(s)
      end if
    end if
  end do
  call Global % Sum_Real(out_rate)

  ! Trapezoidal accumulation.  Euler on this integral is first order in dt and
  ! dominates the result: on Stefan it leaves 4.3e-4 where the trapezoidal rule
  ! leaves 2.4e-5, so an Euler sum measures the quadrature, not the solver.
  if(first) out_prev = out_rate           ! step 1 has no predecessor
  out_cum  = out_cum + 0.5 * (out_rate + out_prev) * Flow % dt
  out_prev = out_rate
  m_s      = m_dom + out_cum

  ! M_0 is the invariant M_s, NOT M_d: this hook first runs at the END of step
  ! one, by which time one step's worth of fluid has already left, so seeding
  ! M_0 from M_d builds that egress into the error as a constant offset.
  if(first) then
    m_zero = m_s
    first  = .false.
  end if

  if(First_Proc()) then
    call File % Append_For_Writing_Ascii('mass-balance.dat', fu)
    write(fu,'(5(2X,E20.12E2))') Time % Get_Time(), m_dom, m_s,  &
                                 m_dom / m_zero, m_s / m_zero - 1.0
    close(fu)
  end if

  end subroutine
