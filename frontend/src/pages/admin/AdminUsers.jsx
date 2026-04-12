import { useState } from 'react'
import AdminPageHeader from '../../components/admin/AdminPageHeader'
import Modal from '../../components/admin/Modal'
import { EmptyState, ErrorState, LoadingSkeleton } from '../../components/admin/EmptyState'
import useFetch from '../../hooks/useFetch'
import { fetchUsers, createUser, updateUser, deactivateUser } from '../../services/api'
import { formatTimestamp } from '../../utils/formatters'

// Mock data for when backend endpoint isn't implemented yet
const MOCK_USERS = [
  { id: 1, name: 'Admin User',  email: 'admin@driftinsights.io', role: 'admin', status: 'active',   created_at: new Date(Date.now() - 86400000 * 30).toISOString(), last_login: new Date(Date.now() - 3600000).toISOString() },
  { id: 2, name: 'ML Engineer', email: 'user@driftinsights.io',  role: 'user',  status: 'active',   created_at: new Date(Date.now() - 86400000 * 14).toISOString(), last_login: new Date(Date.now() - 7200000).toISOString() },
  { id: 3, name: 'Data Analyst',email: 'analyst@example.com',    role: 'user',  status: 'inactive', created_at: new Date(Date.now() - 86400000 * 7).toISOString(),  last_login: null },
]

const EMPTY_FORM = { name: '', email: '', role: 'user', password: '' }

export default function AdminUsers() {
  const { data, loading, error, refetch } = useFetch(fetchUsers, {})
  const users = data?.users?.length ? data.users : MOCK_USERS

  const [modal, setModal]   = useState(null) // null | 'create' | 'edit'
  const [editing, setEditing] = useState(null)
  const [form, setForm]     = useState(EMPTY_FORM)
  const [saving, setSaving] = useState(false)
  const [saveErr, setSaveErr] = useState('')
  const [search, setSearch] = useState('')

  const openCreate = () => { setForm(EMPTY_FORM); setEditing(null); setSaveErr(''); setModal('create') }
  const openEdit   = (u)  => { setForm({ name: u.name, email: u.email, role: u.role, password: '' }); setEditing(u); setSaveErr(''); setModal('edit') }
  const closeModal = ()   => setModal(null)

  const handleSave = async () => {
    if (!form.name || !form.email) { setSaveErr('Name and email are required'); return }
    setSaving(true); setSaveErr('')
    try {
      if (modal === 'create') await createUser(form)
      else await updateUser(editing.id, form)
      refetch(); closeModal()
    } catch (e) {
      setSaveErr(e?.response?.data?.detail || 'Save failed — backend not yet connected')
    }
    setSaving(false)
  }

  const handleDeactivate = async (u) => {
    if (!confirm(`Deactivate ${u.name}?`)) return
    try { await deactivateUser(u.id); refetch() }
    catch { alert('Deactivate failed — backend not yet connected') }
  }

  const filtered = users.filter(u =>
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.email.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div>
      <AdminPageHeader
        title="User Management"
        desc="Create, edit, and deactivate user accounts"
        action={<button className="btn btn-primary" onClick={openCreate}>+ New User</button>}
      />

      {error && <ErrorState message="Could not load users from backend — showing mock data" />}

      {/* Search */}
      <div className="di-card" style={{ padding: '12px 16px', marginBottom: 12 }}>
        <input
          className="admin-input" placeholder="Search by name or email…"
          value={search} onChange={e => setSearch(e.target.value)}
          style={{ maxWidth: 320 }}
        />
      </div>

      <div className="di-card">
        <h3 className="di-card-title">Users <span className="di-badge-sm">{filtered.length} total</span></h3>
        {loading && !users.length ? <LoadingSkeleton /> : filtered.length === 0 ? (
          <EmptyState icon="👤" message="No users found" />
        ) : (
          <div className="di-table-wrap">
            <table className="di-table">
              <thead>
                <tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Created</th><th>Last Login</th><th>Actions</th></tr>
              </thead>
              <tbody>
                {filtered.map(u => (
                  <tr key={u.id}>
                    <td style={{ fontWeight: 500 }}>{u.name}</td>
                    <td className="mono" style={{ fontSize: 11 }}>{u.email}</td>
                    <td>
                      <span className={u.role === 'admin' ? 'role-admin navbar-role-badge' : 'role-user navbar-role-badge'}>
                        {u.role}
                      </span>
                    </td>
                    <td>
                      <span className={u.status === 'active' ? 'user-status-active' : 'user-status-inactive'}>
                        {u.status}
                      </span>
                    </td>
                    <td className="text-muted" style={{ fontSize: 11 }}>{formatTimestamp(u.created_at)}</td>
                    <td className="text-muted" style={{ fontSize: 11 }}>{u.last_login ? formatTimestamp(u.last_login) : '—'}</td>
                    <td>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button className="btn-sm" onClick={() => openEdit(u)}>Edit</button>
                        {u.status === 'active' && (
                          <button className="btn-sm" style={{ color: 'var(--accent-red)' }} onClick={() => handleDeactivate(u)}>Deactivate</button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Create / Edit Modal */}
      <Modal
        open={!!modal}
        onClose={closeModal}
        title={modal === 'create' ? 'Create User' : `Edit — ${editing?.name}`}
        actions={
          <>
            <button className="btn btn-secondary" onClick={closeModal}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
              {saving ? <><span className="spinner" /> Saving…</> : 'Save'}
            </button>
          </>
        }
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {saveErr && <ErrorState message={saveErr} />}
          {[
            { label: 'Full Name', key: 'name', type: 'text', placeholder: 'Jane Smith' },
            { label: 'Email',     key: 'email', type: 'email', placeholder: 'jane@example.com' },
            { label: 'Password',  key: 'password', type: 'password', placeholder: modal === 'edit' ? 'Leave blank to keep current' : 'Min 8 characters' },
          ].map(f => (
            <div key={f.key} className="admin-form-group">
              <label className="admin-label">{f.label}</label>
              <input
                className="admin-input" type={f.type} placeholder={f.placeholder}
                value={form[f.key]} onChange={e => setForm(p => ({ ...p, [f.key]: e.target.value }))}
              />
            </div>
          ))}
          <div className="admin-form-group">
            <label className="admin-label">Role</label>
            <select className="admin-input admin-select" value={form.role} onChange={e => setForm(p => ({ ...p, role: e.target.value }))}>
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>
        </div>
      </Modal>
    </div>
  )
}
