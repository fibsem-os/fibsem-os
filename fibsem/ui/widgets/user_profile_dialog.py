"""Dialogs for managing AutoLamella user profiles."""

import logging
from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import fibsem.config as cfg
from fibsem.applications.autolamella.structures import AutoLamellaUser
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)


class UserProfileDialog(QDialog):
    """Edit a single user's profile fields."""

    def __init__(self, user: Optional[AutoLamellaUser] = None, parent: Optional[QWidget] = None, new_user: bool = False):
        super().__init__(parent)
        self._user = user or AutoLamellaUser()
        self._new_user = new_user
        self.setWindowTitle("New User" if new_user else "Edit User Profile")
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._username = QLineEdit(self._user.username)
        if not self._new_user:
            self._username.setReadOnly(True)
            self._username.setToolTip("OS username — read-only")

        self._name = QLineEdit(self._user.name)
        self._email = QLineEdit(self._user.email)
        self._organisation = QLineEdit(self._user.organization)

        form.addRow("Username", self._username)
        form.addRow("Name", self._name)
        form.addRow("Email", self._email)
        form.addRow("Organisation", self._organisation)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_user(self) -> AutoLamellaUser:
        if self._new_user:
            self._user.username = self._username.text().strip()
        self._user.name = self._name.text().strip()
        self._user.email = self._email.text().strip()
        self._user.organization = self._organisation.text().strip()
        return self._user


class UserManagementDialog(QDialog):
    """List saved users; create, edit, activate, or delete them."""

    def __init__(self, active_user_id: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("User Profiles")
        self.setMinimumWidth(480)
        self.setMinimumHeight(360)
        self._users: List[AutoLamellaUser] = cfg.load_users()
        self._active_user_id = active_user_id or next(
            (u._id for u in self._users if u.is_default), ""
        )
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_edit)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._btn_new = QPushButton("New User")
        self._btn_new.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self._btn_new.clicked.connect(self._on_new)

        self._btn_edit = QPushButton("Edit")
        self._btn_edit.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self._btn_edit.clicked.connect(self._on_edit)

        self._btn_activate = QPushButton("Set Active")
        self._btn_activate.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self._btn_activate.clicked.connect(self._on_activate)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self._btn_delete.clicked.connect(self._on_delete)

        for btn in (self._btn_new, self._btn_edit, self._btn_activate, self._btn_delete):
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_list(self):
        self._list.clear()
        for user in self._users:
            label = user.name or user.username
            if user._id == self._active_user_id:
                label = f"★  {label}"
            item = QListWidgetItem(label)
            item.setData(256, user._id)
            self._list.addItem(item)

    def _selected_user(self) -> Optional[AutoLamellaUser]:
        item = self._list.currentItem()
        if item is None:
            return None
        uid = item.data(256)
        return next((u for u in self._users if u._id == uid), None)

    def _on_new(self):
        dlg = UserProfileDialog(AutoLamellaUser(), self, new_user=True)
        if dlg.exec_() == QDialog.Accepted:
            user = dlg.get_user()
            self._users.append(user)
            if not self._active_user_id:
                user.is_default = True
                self._active_user_id = user._id
            self._refresh_list()

    def _on_edit(self):
        user = self._selected_user()
        if user is None:
            return
        dlg = UserProfileDialog(user, self)
        if dlg.exec_() == QDialog.Accepted:
            updated = dlg.get_user()
            idx = next(i for i, u in enumerate(self._users) if u._id == updated._id)
            self._users[idx] = updated
            self._refresh_list()

    def _on_activate(self):
        user = self._selected_user()
        if user is None:
            return
        for u in self._users:
            u.is_default = False
        user.is_default = True
        self._active_user_id = user._id
        self._refresh_list()

    def _on_delete(self):
        user = self._selected_user()
        if user is None:
            return
        if len(self._users) == 1:
            QMessageBox.warning(self, "Cannot Delete", "At least one user must exist.")
            return
        reply = QMessageBox.question(
            self,
            "Delete User",
            f"Delete user '{user.name or user.username}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._users = [u for u in self._users if u._id != user._id]
        if self._active_user_id == user._id:
            self._active_user_id = self._users[0]._id
            self._users[0].is_default = True
        self._refresh_list()

    def _on_accept(self):
        cfg.save_users(self._users)
        self.accept()

    def get_active_user(self) -> AutoLamellaUser:
        user = next((u for u in self._users if u._id == self._active_user_id), None)
        return user or (self._users[0] if self._users else AutoLamellaUser.from_environment())
